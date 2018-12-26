#ifndef NEOHOOK_PARAMS_H__
#define NEOHOOK_PARAMS_H__

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

namespace params
{
dealii::ParameterHandler GlobPrm;

void DeclareParams() {
  GlobPrm.declare_entry("fem polynomial order","1",
      dealii::Patterns::Integer(),"");
  GlobPrm.declare_entry("domain geometry hyper shape", "rectangle",
      dealii::Patterns::Selection("rectangle"), "");
  GlobPrm.declare_entry("solver type", "direct",
      dealii::Patterns::Selection("direct|iterative"), "");
  GlobPrm.declare_entry("dimension", "3",
      dealii::Patterns::Integer(), "problem dimension")
  GlobPrm.declare_entry("number of cells", "10,10,10",
      dealii::Patterns::List(dealii::Patterns::Integer()), "");
  GlobPrm.declare_entry("side lengths", "1.0, 1.0, 1.0",
      dealii::Patterns::List(dealii::Patterns::Double()), "");
  GlobPrm.declare_entry("uniform refinements", "2",
      dealii::Patterns::Integer(), "");
}

/*!
 This function returns a static const reference to the params::GlobPrm, which
 is a dealii::ParameterHandler object.
 */
static const dealii::ParameterHandler & GetPrm() {
  return GlobPrm;
}

template <typename T>
T GetParam (const std::string & str);

template <>
std::string GetParam<std::string> (const std::string & str) {
  return GlobPrm.get(str);
}

template <>
int GetParam<int> (const std::string & str) {
  return GlobPrm.get_integer(str);
}

template <>
double GetParam<double> (const std::string & str) {
  return GlobPrm.get_double(str);
}

template <>
bool GetParam<bool> (const std::string & str) {
  return GlobPrm.get_bool(str);
}

template <>
std::vector<double> GetParam<std::vector<double>> (const std::string & str) {
  return dealii::Utilities::string_to_double(
      dealii::Utilities::split_string_list(GlobPrm.get(str)));
}

template <>
std::vector<int> GetParam<std::vector<int>> (const std::string & str) {
  return dealii::Utilities::string_to_int(
      dealii::Utilities::split_string_list(GlobPrm.get(str)));
}
}

#endif // NEOHOOK_PARAMS_H__
