# - Find hep-ga library
# Find the native hep-ga includes
# This module defines
#  HEPGA_INCLUDE_DIR, where to find hep/ga.hpp, etc.
#  HEPGA_FOUND, If false, do not try to use HEPGA.
# also defined, but not for general use are

find_path(HEPGA_INCLUDE_DIR hep/ga.hpp HINTS ../hep-ga/include/)

# handle the QUIETLY and REQUIRED arguments and set HEPGA_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(hep-ga  DEFAULT_MSG HEPGA_INCLUDE_DIR)

MARK_AS_ADVANCED(HEPGA_INCLUDE_DIR)
