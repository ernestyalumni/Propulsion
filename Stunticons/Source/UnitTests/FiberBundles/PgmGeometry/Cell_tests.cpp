#include "FiberBundles/PgmGeometry/Cell.h"
#include "gtest/gtest.h"

#include <optional>

using FiberBundles::PgmGeometry::Cell;

namespace GoogleUnitTests
{

namespace FiberBundles
{

namespace PgmGeometry
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CellTests, DefaultConstructible)
{
  Cell cell {};

  EXPECT_TRUE(cell.get_borders().empty());
  EXPECT_EQ(cell.get_neighbor(Cell::BorderPosition::Top), nullptr);
  EXPECT_EQ(cell.get_neighbor(Cell::BorderPosition::Bottom), nullptr);
  EXPECT_EQ(cell.get_neighbor(Cell::BorderPosition::Left), nullptr);
  EXPECT_EQ(cell.get_neighbor(Cell::BorderPosition::Right), nullptr);
  EXPECT_EQ(cell.get_type(), Cell::CellType::Default);
  EXPECT_EQ(cell.i(), 0);
  EXPECT_EQ(cell.j(), 0);
  EXPECT_FALSE(cell.get_id().has_value());
}

} // namespace PgmGeometry

} // namespace FiberBundles

} // namespace GoogleUnitTests