#include "app.hpp"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options.hpp>

namespace op = boost::program_options;

int main(int argc, char* argv[]) {

    int width = 200;
    int height = 200;
    bool gpu = false;

    op::options_description desc("Options");
    desc.add_options()
        ("w", op::value<int>(&width), "screen width in pixels")
        ("h", op::value<int>(&height), "screen height in pixels")
        ("gpu", op::value<bool>(&gpu), "use gpu")
    ;

    op::variables_map vm;
    op::store(op::parse_command_line(argc, argv, desc), vm);
    op::notify(vm);

    App* app = new App(width, height, gpu);
    app->run();
    return 0;
}
