# EDGAR

GO playing robot arm

## One-time setup

All workflows are wrapped in a taskfile.

Refresh calibration when something moves:

```
task calibrate-camera                    # re-pick if USB devices reshuffled
task calibrate-board                     # help the camera find the go board
