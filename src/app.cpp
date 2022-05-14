#include "app.hpp"
#include "cpu_solver.hpp"
#include "slider.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>

App::App(int w, int h, bool gpu) : gridHeight(h), runningSimulation(false) {
    simWidth = w;
    sidebarWidth = 240;
    gridWidth = w + sidebarWidth;
    gridHeight = std::max(h, 240);
    window = new sf::RenderWindow(sf::VideoMode(gridWidth, gridHeight), "SmokeSim");
    simulation = new FluidSim(simWidth, gridHeight, gpu);
    smokeTexture.create(gridWidth, gridHeight);


    // sidebar
    sidebar = sf::RectangleShape(sf::Vector2f(sidebarWidth, gridHeight));
    sidebar.setFillColor(sf::Color(220, 220, 220));
    sidebar.setPosition(gridWidth-sidebarWidth, 0);

    // init sliders
    int sHeight = 30;
    int sWidth = sidebarWidth - 20;
    sf::Vector2f sPos(sidebar.getPosition().x+(sidebarWidth-sWidth)/2, 10);

    viscSlider.initialize(sWidth, sHeight, (int)sPos.x, (int)sPos.y, "Viscosity", 0.0, 0.1, 1.0, 1.0);
    tempSlider.initialize(sWidth, sHeight, (int)sPos.x, (int)sPos.y + 50, "Temperature", 0.0, 0.1, 5.0, 1.0);
    timeSlider.initialize(sWidth, sHeight, (int)sPos.x, (int)sPos.y + 100, "Time Delta", 0.1, 0.1, 5.0, 1.0);
    sizeSlider.initialize(sWidth, sHeight, (int)sPos.x, (int)sPos.y + 150, "Dye Size", 1.0, 0.1, 5.0, 3.0);
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
                simulation->reset();
            } else if (event.key.code == sf::Keyboard::Space) {
                runningSimulation = !runningSimulation;
            } else if (event.key.code == sf::Keyboard::Num1) {
                simulation->changeColor(SmokeColor::WHITE);
            } else if (event.key.code == sf::Keyboard::Num2) {
                simulation->changeColor(SmokeColor::RED);
            } else if (event.key.code == sf::Keyboard::Num3) {
                simulation->changeColor(SmokeColor::GREEN);
            } else if (event.key.code == sf::Keyboard::Num4) {
                simulation->changeColor(SmokeColor::BLUE);
            } else if (event.key.code == sf::Keyboard::B) {
                addBounds = true;
            }
            break;
        }
        case (sf::Event::MouseButtonPressed): {
            int x = event.mouseButton.x;
            int y = event.mouseButton.y;

            if (event.mouseButton.button == sf::Mouse::Left) {
                if (x < simWidth && y < gridHeight)
                    simulation->addDensity(x, y);
                viscSlider.onMousePressed(x, y);
                timeSlider.onMousePressed(x, y);
                sizeSlider.onMousePressed(x, y);
                tempSlider.onMousePressed(x, y);
                updateSliders();
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
                if (currX < simWidth && currY < gridHeight) {
                    if (addBounds) {
                        simulation->addBoundary(currX, currY);
                    } else {
                        simulation->addDensity(currX, currY);
                    }

                }
                viscSlider.onMouseMoved(currX, currY);
                tempSlider.onMouseMoved(currX, currY);
                sizeSlider.onMouseMoved(currX, currY);
                timeSlider.onMouseMoved(currX, currY);
                updateSliders();
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

void App::updateSliders() {
    simulation->viscosity = viscSlider.getValue();
    simulation->timeDelta = timeSlider.getValue();
    simulation->tempDelta = tempSlider.getValue();
    simulation->smokeSize = sizeSlider.getValue();
}

void App::update() {
    simulation->updateSimulation();
}

// draw sprite containing smoke to screen
void App::draw() {
    // update texture of sprite using simulation color data
    sf::Image smokeImage;
    smokeImage.create(simWidth, gridHeight, simulation->denseRGBA);
    smokeTexture.loadFromImage(smokeImage);
    smokeSprite.setTexture(smokeTexture);

    window->draw(smokeSprite);
    window->draw(sidebar);

    viscSlider.draw(window);
    tempSlider.draw(window);
    timeSlider.draw(window);
    sizeSlider.draw(window);

    window->display();
}

