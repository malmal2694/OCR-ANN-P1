#include <Magick++.h> 
#include <iostream> 
#include <unistd.h>
#include <string>
#include <bits/stdc++.h>
#include <map>

using namespace std; 
using namespace Magick; 

const map<string, MagickCore::MorphologyMethod> morphology_methods = {
  {"Close", MagickCore::MorphologyMethod::CloseMorphology},
  {"Erode", MagickCore::MorphologyMethod::ErodeMorphology}, 
  {"Dilate", MagickCore::MorphologyMethod::DilateMorphology}
};
const map<string, MagickCore::KernelInfoType> morphology_kernels = {
  {"Rectangle", MagickCore::RectangleKernel},
  {"Plus", MagickCore::PlusKernel}
};

map <string, string> parse_args(int argc, char* argv[]){
  map <string, string> args_vals_pair;
  for(auto i=0; i < argc; i++){
    if(*(argv + i)[0] == '-')
      // Convert C-style string to C++-style string
      args_vals_pair.insert({string(*(argv + i)), string(*(argv + i + 1))});
  }
  return args_vals_pair;
};

vector<string> split_text(string text, char separator){
  vector<string> sep_list;
  stringstream ss(text);

  while (ss.good()) {
      string substr;
      getline(ss, substr, separator);
      sep_list.push_back(substr);
  }
  return sep_list;
}

int main(int argc,char* argv[]) 
{
  InitializeMagick(*argv);
  // Initial variables
  char pointsize = 35;
  string font = "";
  string background_path;
  string out_name = "test.jpg";
  string size_img = "1000x1000";
  // If this var be true, the morphology method will apply on the image
  auto set_morph = false;
  struct{
  MagickCore::MorphologyMethod method;
  MagickCore::KernelInfoType kernel;
  string size;
  }morph;
  struct{
  // Coordinate origin is up-left side
  string txt = "Hello";
  double pos_x = 0;
  double pos_y = 0;
  }text;
  struct{
  double brightness = 100;
  double saturation = 100;
  double hue = 100;
  }bsh_vals;// Values of brightness, saturation, and hue
  struct{
    double radious = 0;
    double sigma = 0;
  }blur;

  // Get input parameters, parse them, and set defined varaibles to this values.
  auto parsed_args = parse_args(argc, argv); // Parse arg/val pairs and store to this var
  // If user sets the pointsize argument, set the var with this value
  if (parsed_args.find("-pointsize") != parsed_args.end())
    pointsize = stoi(parsed_args["-pointsize"]);
  if (parsed_args.find("-font") != parsed_args.end())
    font = parsed_args["-font"];
  if (parsed_args.find("-outname") != parsed_args.end())
    out_name = parsed_args["-outname"];
  if (parsed_args.find("-size") != parsed_args.end())
    size_img = parsed_args["-size"].c_str();    
  if (parsed_args.find("-blur") != parsed_args.end()){
    const vector<string> temp_blur = split_text(parsed_args["-blur"], '-');
    blur.radious = stod(temp_blur[0]);
    blur.sigma = stod(temp_blur[1]);
  }
  if (parsed_args.find("-morphology") != parsed_args.end()){
    const vector<string> temp_morph = split_text(parsed_args["-morphology"], '-');
    // If user entered the correct morphology format, then apply morphology on the image
    if(temp_morph.size() == 3) set_morph = true;
    morph.method = morphology_methods.at(temp_morph[0]);
    morph.kernel = morphology_kernels.at(temp_morph[1]);
    morph.size = temp_morph[2];
  }
  if (parsed_args.find("-text") != parsed_args.end())
    text.txt = parsed_args["-text"];
  // -pos argument is the coordination of the text
  // Note that the coordinate origin is the up-left side
  if (parsed_args.find("-pos-xy") != parsed_args.end()){
    const vector<string> temp_pos = split_text(parsed_args["-pos-xy"], '-');
    text.pos_x = stod(temp_pos[0]);
    text.pos_y = stod(temp_pos[1]);
  }
  if (parsed_args.find("-bsh") != parsed_args.end()){
    const vector<string> temp_bsh = split_text(parsed_args["-bsh"], '-');
    bsh_vals.brightness = stod(temp_bsh[0]);
    bsh_vals.saturation = stod(temp_bsh[1]);
    bsh_vals.hue = stod(temp_bsh[2]);
  }
  if (parsed_args.find("-background") != parsed_args.end())
    background_path = parsed_args["-background"];

  // Print the key/value pairs user entered
  // for(const auto &a: parsed_args)
  //   cout << a.first << ':' << a.second << '\n';
  try {
    // Create an empty image
    Image image(size_img.c_str(), "white");
    // Background image to compose with the image(text)
    Image texture_img(size_img.c_str(), "white");
    if(background_path != "")
      texture_img.texture(Image(background_path));

    image.resolutionUnits(PixelsPerInchResolution); 
    image.fontPointsize(pointsize);
    image.font(font);
    image.draw(DrawableText(text.pos_x, text.pos_y, text.txt));
    // If user sets mophology parameter, apply the morphology on the image
    if(set_morph)
      image.morphology(morph.method, morph.kernel, morph.size);
    // Composite text and background image
    image.composite(
      texture_img, MagickCore::GravityType::SouthGravity, 
      MagickCore::CompositeOperator::MultiplyCompositeOp
    );
    image.modulate(bsh_vals.brightness, bsh_vals.saturation, bsh_vals.hue);
    image.blur(blur.radious, blur.sigma);
    image.write(out_name);
  } 
  catch( Exception &error_ ) 
    { 
      cout << "Caught exception: " << error_.what() << endl; 
      return 1; 
    } 
  return 0; 
}