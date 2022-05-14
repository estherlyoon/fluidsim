#ifndef __APP_HPP__
#define __APP_HPP__

#include "fluidsim.hpp"
#include "slider.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics.hpp>
//#include <libavformat/avformat.h>
//#include <libavcodec/avcodec.h>
#include <vector>

class App {

public:
    sf::RenderWindow* window;
    sf::Sprite smokeSprite;
    sf::Texture smokeTexture;
    FluidSim* simulation;

    int gridWidth;
    int gridHeight;
    int simWidth;
    int sidebarWidth;
    bool runningSimulation;
    bool addBounds;

    // sidebar elements
    sf::RectangleShape sidebar;
    Slider viscSlider;
    Slider timeSlider;
    Slider tempSlider;
    Slider sizeSlider;

    App(int w, int h, bool gpu);

    ~App();

    void run();
    void update();
    void updateSliders();
    void draw();
    void event_handler(sf::Event const& event);
    void checkSliders();
};


#endif
