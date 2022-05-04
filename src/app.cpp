#include "app.hpp"
#include <SFML/Graphics.hpp>
#include <iostream>

App::App() {
    gridWidth = 800;
    gridHeight = 600;
    window = new sf::RenderWindow(sf::VideoMode(gridWidth, gridHeight), "SmokeSim");
    simulation = new FluidSim(gridWidth, gridHeight);
    smokeTexture.create(gridWidth, gridHeight);
}

App::~App() {
}

void App::run() {

    while (window->isOpen()) {
        // TODO: poll events (smoke placement, clicking, dragging)

        update();
        window->clear(sf::Color::Red);

        draw();
    }
}

void App::update() {

}

// draw sprite containing smoke to screen
void App::draw() {
    // update texture of sprite using simulation color data
    /* smokeTexture.update(simulation->RGBA, gridWidth, gridHeight, 0, 0); */
    sf::Image smokeImage;
    smokeImage.create(gridWidth, gridHeight, simulation->RGBA);
    smokeTexture.loadFromImage(smokeImage);
    smokeSprite.setTexture(smokeTexture);

    // TODO?
    /* sprite.setScale(gridWidth, gridHeight); */

    window->draw(smokeSprite);
    window->display();
}
