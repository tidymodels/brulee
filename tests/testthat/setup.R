# Keep the TabICL attach hook (R/aaa.R) from downloading weights over the network
# during the package's own tests and R CMD check. End users still get the
# automatic download; the package's checks opt out.
options(brulee.tabicl_autodownload = FALSE)
