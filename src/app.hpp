#ifndef __APP_HPP__
#define __APP_HPP__

#include <SFML/Graphics/RenderWindow.hpp>

class App {

public:
    sf::RenderWindow* window;

    App();

    ~App();

    void run();
    void update();
};


#endif
