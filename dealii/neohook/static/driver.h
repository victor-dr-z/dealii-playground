#ifndef NEOHOOK_DRIVER_H__
#define NEOHOOK_DRIVER_H__

class Driver {
 public:
  Driver();

  void Run();

 private:
  void SetupSystem();
  void MakeGrid();
  void Solve();
  void OutputResult();
};


#endif // NEOHOOK_DRIVER_H__