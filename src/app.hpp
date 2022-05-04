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

    unsigned int gridWidth;
    unsigned int gridHeight;

    App();

    ~App();

    void run();
    void update();
    void draw();
    void uint8ToImage();
};


#endif
