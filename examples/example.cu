    struct GS_OUTPUT_FLUIDSIM {   
        // Index of the current grid cell (i,j,k in [0,gridSize] range)    
        float3 cellIndex : TEXCOORD0;   
        // Texture coordinates (x,y,z in [0,1] range) for the    
        // current grid cell and its immediate neighbors    
        float3 CENTERCELL : TEXCOORD1;   
        float3 LEFTCELL   : TEXCOORD2;
        float3 RIGHTCELL  : TEXCOORD3;
        float3 BOTTOMCELL : TEXCOORD4;
        float3 TOPCELL    : TEXCOORD5;
        float3 DOWNCELL   : TEXCOORD6;
        float3 UPCELL     : TEXCOORD7;
        float4 pos        : SV_Position; 
        // 2D slice vertex in    
        // homogeneous clip space    
        uint RTIndex    : SV_RenderTargetArrayIndex;  
    // Specifies    // destination slice 
    }; 

    float3 cellIndex2TexCoord(float3 index) {   // Convert a value in the range [0,gridSize] to one in the range [0,1].    
        return float3(index.x / textureWidth, index.y / textureHeight, (index.z+0.5) / textureDepth); 
    } 

    float4 PS_ADVECT_VEL(GS_OUTPUT_FLUIDSIM in, Texture3D velocity) : SV_Target {   
        float3 pos = in.cellIndex;   
        float3 cellVelocity = velocity.Sample(samPointClamp, in.CENTERCELL).xyz;   
        pos -= timeStep * cellVelocity;   
        pos = cellIndex2TexCoord(pos);   
        return velocity.Sample(samLinear, pos); 
    } 

    float PS_DIVERGENCE(GS_OUTPUT_FLUIDSIM in, Texture3D velocity) : SV_Target {   
        // Get velocity values from neighboring cells.    
        float4 fieldL = velocity.Sample(samPointClamp, in.LEFTCELL);   
        float4 fieldR = velocity.Sample(samPointClamp, in.RIGHTCELL);   
        float4 fieldB = velocity.Sample(samPointClamp, in.BOTTOMCELL);   
        float4 fieldT = velocity.Sample(samPointClamp, in.TOPCELL);   
        float4 fieldD = velocity.Sample(samPointClamp, in.DOWNCELL);   
        float4 fieldU = velocity.Sample(samPointClamp, in.UPCELL);   
        // Compute the velocity's divergence using central differences.    
        float divergence =  0.5 * ((fieldR.x - fieldL.x)+(fieldT.y - fieldB.y)+(fieldU.z - fieldD.z));   
        return divergence; 
    } 

    float PS_JACOBI(GS_OUTPUT_FLUIDSIM in, Texture3D pressure, Texture3D divergence) : SV_Target {   
        // Get the divergence at the current cell.    
        float dC = divergence.Sample(samPointClamp, in.CENTERCELL);
        // Get pressure values from neighboring cells.
        float pL = pressure.Sample(samPointClamp, in.LEFTCELL);
        float pR = pressure.Sample(samPointClamp, in.RIGHTCELL);
        float pB = pressure.Sample(samPointClamp, in.BOTTOMCELL);   
        float pT = pressure.Sample(samPointClamp, in.TOPCELL);   
        float pD = pressure.Sample(samPointClamp, in.DOWNCELL);   
        float pU = pressure.Sample(samPointClamp, in.UPCELL);   
        // Compute the new pressure value for the center cell.    
        return(pL + pR + pB + pT + pU + pD - dC) / 6.0; 
    } 

    float4 PS_PROJECT(GS_OUTPUT_FLUIDSIM in, Texture3D pressure, Texture3D velocity): SV_Target {   
        // Compute the gradient of pressure at the current cell by
        // taking central differences of neighboring pressure values.
        float pL = pressure.Sample(samPointClamp, in.LEFTCELL);
        float pR = pressure.Sample(samPointClamp, in.RIGHTCELL);
        float pB = pressure.Sample(samPointClamp, in.BOTTOMCELL);
        float pT = pressure.Sample(samPointClamp, in.TOPCELL);  
        float pD = pressure.Sample(samPointClamp, in.DOWNCELL);  
        float pU = pressure.Sample(samPointClamp, in.UPCELL);  
        float3 gradP = 0.5*float3(pR - pL, pT - pB, pU - pD);   
        // Project the velocity onto its divergence-free component by   
        // subtracting the gradient of pressure.    
        float3 vOld = velocity.Sample(samPointClamp, in.texcoords);  
        float3 vNew = vOld - gradP;   return float4(vNew, 0); 
    } 

    // MacCormack Advection Scheme

    float4 PS_ADVECT_MACCORMACK(GS_OUTPUT_FLUIDSIM in, float timestep) : SV_Target {   
        // Trace back along the initial characteristic - we'll use    
        // values near this semi-Lagrangian "particle" to clamp our    
        // final advected value.    
        float3 cellVelocity = velocity.Sample(samPointClamp, in.CENTERCELL).xyz; 
        float3 npos = in.cellIndex - timestep * cellVelocity; 
        // Find the cell corner closest to the "particle" and compute the    
        // texture coordinate corresponding to that location. 
        npos = floor(npos + float3(0.5f, 0.5f, 0.5f)); 
        npos = cellIndex2TexCoord(npos); 
        // Get the values of nodes that contribute to the interpolated value.    
        // Texel centers will be a half-texel away from the cell corner.   
        float3 ht = float3(0.5f / textureWidth, 0.5f / textureHeight, 0.5f / textureDepth); 
        float4 nodeValues[8]; 
        nodeValues[0] = phi_n.Sample(samPointClamp, npos + float3(-ht.x, -ht.y, -ht.z)); 
        nodeValues[1] = phi_n.Sample(samPointClamp, npos + float3(-ht.x, -ht.y, ht.z)); 
        nodeValues[2] = phi_n.Sample(samPointClamp, npos + float3(-ht.x, ht.y, -ht.z)); 
        nodeValues[3] = phi_n.Sample(samPointClamp, npos + float3(-ht.x, ht.y, ht.z)); 
        nodeValues[4] = phi_n.Sample(samPointClamp, npos + float3(ht.x, -ht.y, -ht.z)); 
        nodeValues[5] = phi_n.Sample(samPointClamp, npos + float3(ht.x, -ht.y, ht.z)); 
        nodeValues[6] = phi_n.Sample(samPointClamp, npos + float3(ht.x, ht.y, -ht.z)); 
        nodeValues[7] = phi_n.Sample(samPointClamp, npos + float3(ht.x, ht.y, ht.z)); 

        // Determine a valid range for the result.    
        float4 phiMin = min(min(min(min(min(min(min(nodeValues[0], nodeValues [1]), nodeValues [2]), nodeValues [3]), nodeValues[4]), nodeValues [5]), nodeValues [6]), nodeValues [7]); 
        float4 phiMax = max(max(max(max(max(max(max(nodeValues[0], nodeValues [1]), nodeValues [2]), nodeValues [3]), nodeValues[4]), nodeValues [5]), nodeValues [6]), nodeValues [7]); 

        // Perform final advection, combining values from intermediate    
        // advection steps.    
        float4 r = phi_n_1_hat.Sample(samLinear, nposTC) + 0.5 * (phi_n.Sample(samPointClamp, in.CENTERCELL) - phi_n_hat.Sample(samPointClamp, in.CENTERCELL)); 
        // Clamp result to the desired range. 
        r = max(min(r, phiMax), phiMin); return r; 
    } 


