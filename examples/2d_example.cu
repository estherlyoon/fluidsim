 // Maintain scalar fields for velocity, pressure, color ("dye") (three scalar fields), and temperature + density
 // Note: for color, must also advect; potentially need to diffuse too unless numerical error in advection handles diffusion
 // Temperature and density scalar required for smoke
    // advect density normally
    // modify buoyant force to account for gravitational pull on dense smoke (eq 22)

 // Apply the first 3 operators in Equation 12.
 u = advect(u); 
 u = diffuse(u); 
 u = addForces(u); 

 // Now apply the projection operator to the result. 
 p = computePressure(u); 
 u = subtractPressureGradient(u, p); 


// probably need tmp variables because these can't be done in-place

// advection
void advect(float2 coords : WPOS, // grid coordinates     
    out float4 xNew : COLOR, // advected qty     
    uniform float timestep,           
    uniform float rdx, // 1 / grid scale
    uniform samplerRECT u, // input velocity     
    uniform samplerRECT x) // qty to advect (could be a velocity or another quantity like dye concentration)
    {   
    // follow the velocity field "back in time"     
    float2 pos = coords - timestep * rdx * f2texRECT(u, coords);
    // interpolate 4 texels closest to pos and write to the output fragment   
    xNew = f4texRECTbilerp(x, pos); 
}

// viscous diffusion (solve poisson dist)
// run 20 to 50 iterations (more is better)
// set alpha to x^2/t, beta to 1/(4+x^2/t)
// set x and b to the velocity texture (u) in the case of diffusion
void jacobi(half2 coords : WPOS,   // grid coordinates     
    out half4 xNew : COLOR,  // result 
    uniform half alpha, 
    uniform half rBeta, 
    // reciprocal beta
    uniform samplerRECT x,   // x vector (Ax = b)  
    uniform samplerRECT b)   // b vector (Ax = b) 
    {   
    // left, right, bottom, and top x samples  
    half4 xL = h4texRECT(x, coords - half2(1, 0));  
    half4 xR = h4texRECT(x, coords + half2(1, 0)); 
    half4 xB = h4texRECT(x, coords - half2(0, 1));  
    half4 xT = h4texRECT(x, coords + half2(0, 1));
    // b sample, from center 
    half4 bC = h4texRECT(b, coords);
    // evaluate Jacobi iteration  
    xNew = (xL + xR + xB + xT + alpha * bC) * rBeta;
}

// force application
// simple equation based on mouse drag
// c = quantity to add to color of fragment
// F = force computed from direction and length of mouse drag
// r = desired impulse radius
// (x, y) = fragment position
// (xp, yp) = impulse (click) position in window coordinates
c = F(deltaT)*exp(((x-x_p)^2 + (y-y_p)^2)/r)

// projection step

// divergence
 void divergence(half2 coords : WPOS,   // grid coordinates
    out half4 div : COLOR,  // divergence-- use as b input for Jacobi program (set x to pressure texture, initially all 0s)
    uniform half halfrdx,   // 0.5 / gridscale    
    uniform samplerRECT w)  // intermediate velocity vector field 
    {   

    half4 wL = h4texRECT(w, coords - half2(1, 0)); 
    half4 wR = h4texRECT(w, coords + half2(1, 0));
    half4 wB = h4texRECT(w, coords - half2(0, 1));  
    half4 wT = h4texRECT(w, coords + half2(0, 1));
    div = halfrdx * ((wR.x - wL.x) + (wT.y - wB.y)); 
}

// use 40-80 Jacobi iterations
// then, bind the pressure texture field to p in the gradient subtraction program
// which computes gradient of p then subtracts it from intermediate velocity field texture (w):
void gradient(half2 coords : WPOS, // grid coordinates
    out half4 uNew : COLOR,  // new velocity  
    uniform half halfrdx,    // 0.5 / gridscale   
    uniform samplerRECT p,   // pressure    
    uniform samplerRECT w)   // velocity 
    {   

    half pL = h1texRECT(p, coords - half2(1, 0));   
    half pR = h1texRECT(p, coords + half2(1, 0));   
    half pB = h1texRECT(p, coords - half2(0, 1));   
    half pT = h1texRECT(p, coords + half2(0, 1));
    uNew = h4texRECT(w, coords);   
    uNew.xy -= halfrdx * half2(pR - pL, pT - pB); 
}


// AFTER: boundary conditions
