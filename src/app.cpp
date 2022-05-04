#include "app.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>

App::App() : gridWidth(800), gridHeight(600), runningSimulation(false) {
    window = new sf::RenderWindow(sf::VideoMode(gridWidth, gridHeight), "SmokeSim");
    simulation = new FluidSim(gridWidth, gridHeight);
    smokeTexture.create(gridWidth, gridHeight);
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
        draw();
    }
}

void App::event_handler(sf::Event const& event) {
    switch (event.type) {
        case (sf::Event::Closed):
            window->close();
            break;
        case (sf::Event::KeyPressed):
            if (event.key.code == sf::Keyboard::R) {
                // TODO reset
            } else if (event.key.code == sf::Keyboard::Space) {
                runningSimulation = true;
            }
            break;
        case (sf::Event::MouseButtonPressed):
            int x = event.mouseButton.x;
            int y = event.mouseButton.y;
            std::cout << "x = " << x << ", y = " << y << std::endl;

            if(event.mouseButton.button == sf::Mouse::Left) {
                for (int i = 0; i < 3; i++) {
                    std::cout << "set at " << y*gridHeight+x << std::endl;
                    simulation->RGBA[(y*gridWidth+x)*4+i] = 255;
                }
            }
            break;
    }
}

void App::update() {

}

// draw sprite containing smoke to screen
void App::draw() {
    // update texture of sprite using simulation color data
    sf::Image smokeImage;
    smokeImage.create(gridWidth, gridHeight, simulation->RGBA);
    smokeTexture.loadFromImage(smokeImage);
    smokeSprite.setTexture(smokeTexture);
    /* sprite.setScale(gridWidth, gridHeight); */

    window->draw(smokeSprite);
    window->display();
}
