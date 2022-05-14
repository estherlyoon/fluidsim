#include "slider.hpp"
#include "common.cuh"

#include <string>

Slider::Slider() {
    m_size = sf::Vector2f(50, 100);
}

Slider::~Slider() {

}

void Slider::initialize(int width, int height, int xPos, int yPos, const std::string label, float start, float step, float max, float initVal) {
    m_value = initVal;
    m_step = step;
    m_minVal = start;
    m_maxVal = max;
    m_size = sf::Vector2f(width, height);
    m_name = label;
    m_wDiff = 10;
    m_hDiff = 20;

    if (!m_font.loadFromFile("../fonts/arial.ttf")) {
        printf("Could not load slider bar font!\n");
        exit(-1);
    }

    // TODO change start of target to middle
    setPosition(xPos, yPos);
    setSize(width, height);
    setText(label);

    float handleX = m_handle.getPosition().x + initVal/m_maxVal * (m_bar.getSize().x-m_wDiff);
    m_handle.setPosition(sf::Vector2f(handleX, m_handle.getPosition().y));
}
 
void Slider::setPosition(const sf::Vector2f& pos) {
    m_pos = pos;
}

void Slider::setPosition(float x, float y) {
    m_pos.x = x;
    m_pos.y = y;
}

void Slider::setText(const std::string& label) {
    std::string text = label + ": " + ftos(m_value);
    m_label.setString(text);
    m_label.setFont(m_font);
    m_label.setCharacterSize(m_size.y/2); // in pixels
    m_label.setStyle(sf::Text::Bold); 
    m_label.setFillColor(sf::Color::Black);
}

void Slider::setSize(float width, float height) {
    m_background.setSize(sf::Vector2f(width, height));
    m_background.setPosition(m_pos.x, m_pos.y);
    m_background.setFillColor(sf::Color(140, 140, 140));

    m_bar.setSize(sf::Vector2f(width-2*m_wDiff, height-2*m_hDiff));
    m_bar.setPosition(m_background.getGlobalBounds().left+m_wDiff, m_background.getGlobalBounds().top+m_hDiff);
    m_bar.setFillColor(sf::Color(200, 200, 200));

    m_handle.setSize(sf::Vector2f(m_bar.getSize().y, m_bar.getSize().y));
    m_handle.setPosition(m_bar.getGlobalBounds().left+m_wDiff, m_bar.getGlobalBounds().top+m_hDiff/2);
    m_handle.setFillColor(sf::Color::Black);

    // label
    m_label.setPosition((m_background.getPosition().x + width)/2, m_background.getPosition().y + height);
}

float Slider::getValue() const {
    return m_value;
}

void Slider::setValue(float value) {
    m_value = value;
}

bool Slider::containsPoint(const sf::Vector2f& p) const {
    auto global = m_bar.getGlobalBounds();
    return (global.left + m_wDiff<= p.x && global.left + global.width + m_wDiff/2 >= p.x
        && global.top <= p.y && global.top + global.height >= p.y);
}

void Slider::updateSlider(float x, float y) {
    float newVal = x - m_bar.getGlobalBounds().left - m_wDiff;
    // scale to fit min and max
    newVal = m_minVal + newVal/m_bar.getSize().x * (m_maxVal - m_minVal);
    setValue(newVal);
    sf::Vector2f newPos(x, m_handle.getPosition().y);
    m_handle.setPosition(newPos);

    std::string text = m_name + ": " + ftos(m_value);
    m_label.setString(text);
}

void Slider::onMousePressed(float x, float y) {
    if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && containsPoint(sf::Vector2f(x, y))) {
        updateSlider(x, y);
    }
}

void Slider::onMouseMoved(float x, float y) {
    if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && containsPoint(sf::Vector2f(x, y))) {
        updateSlider(x, y);
    }
}

void Slider::draw(sf::RenderTarget* target) const {
    target->draw(m_background);
    target->draw(m_bar);
    target->draw(m_handle);
    target->draw(m_label);
}
     
void Slider::draw(sf::RenderTarget& target, sf::RenderStates states) const {}
