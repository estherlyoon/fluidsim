#include "app.hpp"
#include "cpu_solver.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>

// TODO parameterize w, h, gpu
App::App() : gridWidth(800), gridHeight(600), runningSimulation(false), gpu(false) {
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
                runningSimulation = !runningSimulation;
            }
            break;
        case (sf::Event::MouseButtonPressed):
            int x = event.mouseButton.x;
            int y = event.mouseButton.y;

            if(event.mouseButton.button == sf::Mouse::Left) {
                for (int i = 0; i < 3; i++) {
                    simulation->RGBA[(y*gridWidth+x)*4+i] = 255;
                }
            }
            break;
    }
}

// TODO pass in time?
void App::update() {
    if (gpu) {

    } else {
        cpu_solver::update(simulation);
    }
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
