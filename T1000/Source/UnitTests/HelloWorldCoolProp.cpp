//------------------------------------------------------------------------------
/// \file HelloWorldCoolProp.cpp
///
/// \ref http://www.coolprop.org/coolprop/HighLevelAPI.html#c-sample-code
//------------------------------------------------------------------------------

#include "CoolProp.h"
#include <iostream>
#include <vector>

int main()
{
  std::cout << CoolProp::PropsSI("T", "P", 101325, "Q", 0, "Water") <<
    std::endl;
  std::cout << CoolProp::PropsSI("T", "P", 101325, "Q", 0, "Nitrogen") <<
    std::endl;

  // First type (slowest, due to most string processing exposed in DLL)
  std::cout <<
    CoolProp::PropsSI("Dmolar","T",298,"P",1e5,"Propane[0.5]&Ethane[0.5]") <<
      std::endl; // Default backend is HEOS
  std::cout <<
    CoolProp::PropsSI("Dmolar","T",298,"P",1e5,"HEOS::Propane[0.5]&Ethane[0.5]")
      << std::endl;
  //std::cout <<
    //CoolProp::PropsSI(
      //"Dmolar","T",298,"P",1e5,"REFPROP:Propane[0.5]&Ethane[0.5]") << std::endl;

  // Vector example
  // If parentheses is not used, segmentation fault (core dump).
  // Seg fault is caused by accessing memory that "doesn't belong to you",
  // error indicating memory corruption or accessing freed block of memory or
  // do write operation on read-only location.
  std::vector<double> z (2, 0.5);
  std::cout << z.size() << std::endl;

  // Second type (C++ only, a bit faster, allows for vector inputs and outputs)
  std::vector<std::string> fluids;
  fluids.push_back("Propane");
  fluids.push_back("Ethane");

  std::vector<std::string> outputs;
  outputs.push_back("Dmolar");

  // cf. https://stackoverflow.com/questions/35285913/when-should-we-use-parenthesis-vs-initializer-syntax-to-initialize-obje
  // You must use parentheses in order to use the constructor that takes 2
  // elements, explicit vector (size_type n, const value_type& val, ...)
  std::vector<double> T (1,298);
  std::vector<double> p (1,1e5);
  std::cout << T.size() << std::endl;
  std::cout << p.size() << std::endl;


  std::cout <<
    CoolProp::PropsSImulti(outputs, "T", T, "P", p, "", fluids, z)[0][0] <<
      std::endl; // Default backend is HEOS


  //return 1;
  return EXIT_SUCCESS;
}