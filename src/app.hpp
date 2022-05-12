#ifndef __APP_HPP__
#define __APP_HPP__

#include "fluidsim.hpp"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics.hpp>
#include <vector>

class App {

public:
    sf::RenderWindow* window;
    sf::Sprite smokeSprite;
    sf::Texture smokeTexture;
    FluidSim* simulation;

    int gridWidth;
    int gridHeight;
    bool runningSimulation;

    App(int w, int h, bool gpu);

    ~App();

    void run();
    void update();
    void draw();
    void event_handler(sf::Event const& event);
};


#endif
