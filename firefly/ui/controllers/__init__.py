"""Python QObject controllers that bridge the FIREFLY backend to the QML UI.

Each controller exposes Properties / Slots / Signals consumed by the QML layer
(see firefly/ui/qml/). They wrap the existing analysis core + worker + QSettings
without changing them — the UI shell is what's being rewritten, not the science.
"""
