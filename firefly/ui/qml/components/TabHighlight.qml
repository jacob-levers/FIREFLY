import QtQuick

// TabHighlight — a single accent pill that SLIDES to the active tab instead of
// each tab popping its own colour. Parent it to the same Item that holds your
// tab Row, BEHIND the labels (declare it before the Row, or give the Row a
// higher z). Point `target` at the active tab's Item.
//
//   Item {
//       TabHighlight { target: tabRow.children[App.currentTab] }
//       Row {
//           id: tabRow
//           Repeater { model: App.tabs; delegate: Rectangle {
//               color: "transparent"            // no per-tab fill any more
//               implicitWidth: lbl.implicitWidth + sc.sp8 * 2; implicitHeight: 30
//               Text { id: lbl; anchors.centerIn: parent; text: modelData
//                      color: App.currentTab === index ? pal.ACC : pal.TXT_MUTED }
//               TapHandler { onTapped: App.setTab(index) }
//           } }
//       }
//   }
Rectangle {
    id: root
    property Item target: null

    x:      target ? target.x : 0
    y:      target ? target.y : 0
    width:  target ? target.width : 0
    height: target ? target.height : 30
    radius: Theme.scale.radiusLg
    color:  Qt.rgba(Theme.palette.ACC.r, Theme.palette.ACC.g, Theme.palette.ACC.b, 0.14)   // ACC @ 14%
    border.width: 1
    border.color: Theme.palette.ACC

    Behavior on x      { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic } }
    Behavior on width  { NumberAnimation { duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic } }
}
