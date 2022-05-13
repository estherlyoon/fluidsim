#include "app.hpp"
#include "cpu_solver.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>

App::App(int w, int h, bool gpu) : gridWidth(w), gridHeight(h), runningSimulation(false) {
    window = new sf::RenderWindow(sf::VideoMode(gridWidth, gridHeight), "SmokeSim");
    simulation = new FluidSim(gridWidth, gridHeight, gpu);
    smokeTexture.create(gridWidth, gridHeight);

    char* filename = "test.avi";
    int rc = initVideo(filename);
    if (rc != 0)
        std::cerr << "Failed to initialize video: " << rc << std::endl;
}

App::~App() {
}

void App::run() {
    while (window->isOpen()) {
        sf::Event e;
        // check for user input
        while (window->pollEvent(e)) {
            event_handler(e);
        }

        if (runningSimulation) {
            update();
        }

        window->clear();
        display();
        draw();
    }

    closeVideo();
}

void App::event_handler(sf::Event const& event) {
    switch (event.type) {
        case (sf::Event::Closed): {
            window->close();
            break;
        }
        case (sf::Event::KeyPressed): {
            if (event.key.code == sf::Keyboard::R) {
                simulation->reset();
            } else if (event.key.code == sf::Keyboard::Space) {
                printf("run sim\n");
                runningSimulation = !runningSimulation;
            } else if (event.key.code == sf::Keyboard::Num1) {
                simulation->changeColor(SmokeColor::WHITE);
            } else if (event.key.code == sf::Keyboard::Num2) {
                simulation->changeColor(SmokeColor::RED);
            } else if (event.key.code == sf::Keyboard::Num3) {
                simulation->changeColor(SmokeColor::GREEN);
            } else if (event.key.code == sf::Keyboard::Num4) {
                simulation->changeColor(SmokeColor::BLUE);
            }
            break;
        }
        case (sf::Event::MouseButtonPressed): {
            int x = event.mouseButton.x;
            int y = event.mouseButton.y;

            if (event.mouseButton.button == sf::Mouse::Left) {
                simulation->addDensity(x, y);
            }

            if (event.mouseButton.button == sf::Mouse::Right) {
                simulation->xPoint = x;
                simulation->yPoint = y;
            }     
            break;
        }
        case (sf::Event::MouseMoved): {
            // continue applying force in drag direction
            int lastX = simulation->xPoint;
            int lastY = simulation->yPoint;
            float currX = static_cast<float>(event.mouseMove.x);
            float currY = static_cast<float>(event.mouseMove.y);
            float xDir = currX - lastX;
            float yDir = currY - lastY;

            // bounds check
            if (currX < 0 || currX >= gridWidth || currY < 0 || currY >= gridHeight
                || lastX < 0 || lastX >= gridWidth || lastY < 0 || lastY >= gridHeight)
                break;

            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                simulation->addDensity(currX, currY);
            }

            if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
                simulation->xPoint = currX;
                simulation->yPoint = currY;
                simulation->addVelocity(currX, currY, xDir, yDir);
            }
            break;
        }
    }
}

void App::update() {
    simulation->updateSimulation();
}

// draw sprite containing smoke to screen
void App::draw() {
    // update texture of sprite using simulation color data
    sf::Image smokeImage;
    smokeImage.create(gridWidth, gridHeight, simulation->denseRGBA);
    smokeTexture.loadFromImage(smokeImage);
    smokeSprite.setTexture(smokeTexture);

    window->draw(smokeSprite);
    window->display();
}

int App::initVideo(char* filename) {
    /* AVCodec* pCodec; */

/*    if(avformat_open_input(&pFormatCtx, filename, NULL, 0)!=0) {*/
/*        fprintf(stderr, "File does not exist!\n");*/
/*        return -1;*/
/*      }*/

/*    if(avformat_find_stream_info(pFormatCtx, NULL)<0) {*/
/*        fprintf(stderr, "Couldn't find stream information!\n");*/
/*        return -1;*/
/*    }*/

/*    av_dump_format(pFormatCtx, 0, filename, 0);*/

/*    videoStream = -1;*/
/*    for(int i = 0; i < (pFormatCtx->nb_streams); i++) {*/
/*        if(pFormatCtx->streams->codec->codec_type == AVMEDIA_TYPE_VIDEO) {*/
/*            videoStream = i;*/
/*            break;*/
/*        }*/
/*    }*/

/*    if(videoStream == -1)*/
/*        return -1;*/

/*    pCodecCtx = pFormatCtx->streams[videoStream]->codec;*/


/*    pCodec = avcodec_find_decoder(pCodecCtx->codec_id);*/
/*    if(pCodec == NULL) {*/
/*        fprintf(stderr, "Unsupported codec!\n");*/
/*        return -1;*/
/*    }*/

/*    if(avcodec_open2(pCodecCtx, pCodec)<0)*/
/*        return -1;*/

/*    iFrameSize = pCodecCtx->width * pCodecCtx->height * 3;*/
/*    pFrame = avcodec_frame_alloc();*/
/*    pFrameRGB = avcodec_alloc_frame();*/

/*    if(pFrameRGB == NULL)*/
/*        return -1;*/

/*    PFrame is prepared to store our video is in YUV format, ie Hue, Saturation and Brightness. Then, we prepare pFrameRGB to store the video in RGB format, with which work SFML.*/

/*    int numBytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);*/
/*    buffer = (uint8_t*)av_malloc(numBytes*sizeof(uint8_t));*/
/*    avpicture_get_size((AVPicture*)pFrameRGB, buffer, AV_PIX_FMT_RGB24,*/
/*                pCodecCtx->width, pCodecCtx->height);*/

/*    With numbytes we get the number of bytes of an image in RGB24 and dimensions of the video and it allocates the buffer with. Then, the buffer is assigned to pFrameRGB.*/

/*    data = new sf::Uint8[pCodecCtx->width * pCodecCtx->height * 4];*/

    return 0;
}

void App::display() {
    /* int frameFinished; */
 
    /* if (av_read_packet(pFormatCtx, &packet) < 0) { */
    /*     closeVideo(); */
    /*     exit(0); */
    /* } */
 
    /* if(packet.stream_index == videoStream) { */
    /*     avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, */
    /*                     packet.data, packet.size); */
 
    /*     if(frameFinished) { */
    /*         sws_scale(pFrameRGB, AV_PIX_FMT_RGB24, */
    /*               (AVPicture*)pFrame, pCodecCtx->pix_fmt, pCodecCtx->width, */
    /*               pCodecCtx->height); */
    /*     } */

    /* int j = 0; */
    /* for(int i = 0; i < iFrameSize; i+=3) { */
    /*   data[j] = pFrameRGB->data[0][i]; */
    /*   data[j+1] = pFrameRGB->data[0][i+1]; */
    /*   data[j+2] = pFrameRGB->data[0][i+2]; */
    /*   data[j+3] = 255; */
    /*   j+=4; */
    /* } */

    /* im_video.create(pCodecCtx->width, pCodecCtx->height, data); */

  /* } */

  //Draw the image on the screen buffer
  /* window->draw(im_video); */
}

void App::closeVideo() {
    //Free the allocated packet av_read_frame
    /* av_packet_unref(&packet); */
    /* // Free the RGB image */
    /* av_free(buffer); */
    /* av_free(pFrameRGB); */
    /* // Free the YUV image */
    /* av_free(pFrame); */
    /* //Close the codec */
    /* avcodec_close(pCodecCtx); */
    /* //Close the video file */
    /* avformat_close_input(&pFormatCtx); */
}
