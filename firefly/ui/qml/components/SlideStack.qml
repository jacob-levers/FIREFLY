import QtQuick

// SlideStack — animated swap container for tab CONTENT (and any A↔B view swap).
// FIREFLY DEFAULT = pure crossfade (slide: 0). Each FIREFLY tab has a different
// structural layout (Import has the left parameter dock; Analysis has no dock
// but a right Conditions panel; Visualise has a left Settings + right Layers
// panel), so a horizontal slide of "one content pane" reads as broken when the
// side panels appear/disappear. Crossfade the body, and let each tab's own
// edge-docked panels slide in from their edge (see EdgePanel pattern below).
// Only set `slide` > 0 if you are swapping views that share an identical frame.
//
//   SlideStack {
//       id: stack
//       index: App.currentTab
//       slide: 0                     // crossfade — the FIREFLY default
//       views: [importView, processView, analysisView, visualiseView]
//   }
//
// Internally it keeps ONE Loader and crossfades on index change.
Item {
    id: root
    property int index: 0
    property var views: []          // array of Component
    property int _prev: index
    property int slide: 0           // px horizontal travel; 0 = pure fade (FIREFLY default)

    Loader {
        id: ld
        anchors.fill: parent
        sourceComponent: root.views.length > root.index ? root.views[root.index] : null
        opacity: 1
        transform: Translate { id: tr; x: 0 }
    }

    onIndexChanged: {
        var dir = index > _prev ? 1 : -1
        _prev = index
        if (Theme.reducedMotion) { ld.opacity = 1; tr.x = 0; return }
        swap.dir = dir
        swap.restart()
    }

    SequentialAnimation {
        id: swap
        property int dir: 1
        // out
        ParallelAnimation {
            NumberAnimation { target: ld; property: "opacity"; to: 0; duration: 90; easing.type: Easing.OutCubic }
            NumberAnimation { target: tr; property: "x"; to: -swap.dir * root.slide; duration: 90; easing.type: Easing.OutCubic }
        }
        // (Loader source already updated by the binding) — snap to the entry offset
        PropertyAction { target: tr; property: "x"; value: swap.dir * root.slide }
        // in
        ParallelAnimation {
            NumberAnimation { target: ld; property: "opacity"; to: 1; duration: 200; easing.type: Easing.OutCubic }
            NumberAnimation { target: tr; property: "x"; to: 0; duration: 200; easing.type: Easing.OutCubic }
        }
    }
}
