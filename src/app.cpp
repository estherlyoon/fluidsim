#include "app.hpp"
#include <SFML/Graphics.hpp>

App::App() {
    window = new sf::RenderWindow(sf::VideoMode(800, 600), "Smoke Simulation");
}

App::~App() {
    delete window;
}

void App::run() {

    while (window->isOpen()) {
        // TODO:
        // later: poll events (clicking, dragging)

        update();
        window->clear(sf::Color::Red);

        // TODO:
        // draw new frame
        window->display();
    }
}

void App::update() {

}

