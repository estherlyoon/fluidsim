#include "app.hpp"
#include "cpu_solver.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>

App::App(int w, int h, bool gpu) : gridWidth(w), gridHeight(h), runningSimulation(false) {
    window = new sf::RenderWindow(sf::VideoMode(gridWidth, gridHeight), "SmokeSim");
    simulation = new FluidSim(gridWidth, gridHeight, gpu);
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
        case (sf::Event::Closed): {
            window->close();
            break;
        }
        case (sf::Event::KeyPressed): {
            if (event.key.code == sf::Keyboard::R) {
                // TODO reset
            } else if (event.key.code == sf::Keyboard::Space) {
                runningSimulation = !runningSimulation;
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
            printf("mouseMoved\n");
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
                printf("addDensity\n");
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
