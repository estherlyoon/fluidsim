#ifndef __SLIDER_HPP__
#define __SLIDER_HPP__

#include <SFML/Graphics.hpp>
#include <string>
#include <functional>

class Slider: public sf::Drawable {

public:
    Slider();
    ~Slider();

    void initialize(int width, int height, int xPos, int yPos, const std::string label, float start, float step, float max, float initVal);

    void setPosition(const sf::Vector2f& pos);
    void setPosition(float x, float y);
    void setText(const std::string& label);
    void setSize(float width, float height);
    const sf::Vector2f& getPosition() const;
    sf::Vector2f getAbsolutePosition() const;

    float getValue() const;
    void setValue(float value);

    bool containsPoint(const sf::Vector2f& point) const;
    void setCallback(std::function<void(void)> callback);

    void onMousePressed(float x, float y);
    void onMouseMoved(float x, float y);
    void draw(sf::RenderTarget* target) const;

private:
    sf::Vector2f m_pos;
    sf::Vector2f m_size;
    sf::RectangleShape m_background;
    sf::RectangleShape m_bar;
    sf::RectangleShape m_handle;

    sf::Font m_font;
    sf::Text m_label;

    std::string m_name;
    float m_value;
    float m_step;
    float m_minVal;
    float m_maxVal;
    float m_wDiff;
    float m_hDiff;

    void updateSlider(float x, float y);
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

};

#endif
