#ifndef FIBER_BUNDLES_CELL_H
#define FIBER_BUNDLES_CELL_H

namespace FiberBundles
{

class Cell
{
  public:

    enum class CellType
    {
      Fluid,
      Outlet,
      Inlet,
      NoslipWall,
      FreeslipWall,
      Default
    };

    Cell(const CellType cell_type);

  private:

    CellType type_;    
};

} // namespace FiberBundles

#endif // FIBER_BUNDLES_CELL_H