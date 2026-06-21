import QtQuick

// EdgePanel — an edge-docked side panel (the Import parameter dock, the Analysis
// "Conditions" panel, the Visualise "Settings" / "Layers" panels). When its tab
// becomes active it SLIDES IN from the edge it docks to + fades. This is what
// makes a FIREFLY tab change read correctly: the body crossfades, and each
// panel arrives from its own side — never a single horizontal slide of the whole
// content (the layout differs per tab, so that would look broken).
//
//   EdgePanel { edge: "left";  active: App.currentTab === 0   // Import dock
//       ParameterSidebar { ... } }
//   EdgePanel { edge: "right"; active: App.currentTab === 2   // Conditions
//       ConditionsPanel { ... } }
//
// Pair with SlideStack (slide: 0) on the body and FadeRise on the inner cards.
Item {
    id: root
    property string edge: "left"     // "left" | "right"
    property bool   active: true
    property int    travel: 18
    property int    duration: 260
    default property alias content: holder.data

    implicitWidth:  holder.childrenRect.width
    implicitHeight: holder.childrenRect.height
    clip: true

    Item {
        id: holder
        width: root.width; height: root.height
        opacity: root.active ? 1 : 0
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : root.duration; easing.type: Easing.OutCubic } }
        transform: Translate {
            x: root.active ? 0 : (root.edge === "right" ? root.travel : -root.travel)
            Behavior on x { NumberAnimation { duration: Theme.reducedMotion ? 0 : root.duration; easing.type: Easing.OutCubic } }
        }
    }
}
