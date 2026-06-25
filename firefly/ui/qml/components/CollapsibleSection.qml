import QtQuick
import QtQuick.Layouts

// Accordion section, styled as a bordered card: a 42-px header (icon-chip +
// title + chevron) over animated-height content, with an accent left-rail and a
// hairline divider when open. Children declared inside land in the content
// column. Used by every parameter sidebar.
Item {
    id: root
    property string title: ""
    property string icon: ""
    property bool expanded: true
    default property alias content: body.data

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale

    width: parent ? parent.width : implicitWidth
    implicitHeight: card.height

    Rectangle {
        id: card
        width: parent.width
        height: col.implicitHeight
        radius: sc.radius2xl                       // 10
        color: pal.PANEL
        border.width: 1
        border.color: (hh.hovered && !root.expanded) ? pal.BORDER_HI : pal.BORDER
        clip: true
        Behavior on border.color { ColorAnimation { duration: Theme.reducedMotion ? 0 : 120 } }

        // accent left-rail (open only); inset past the rounded corners so it
        // reads as a clean vertical mark rather than poking the radius.
        Rectangle {
            visible: root.expanded
            anchors {
                left: parent.left; top: parent.top; bottom: parent.bottom
                topMargin: sc.radius2xl; bottomMargin: sc.radius2xl
            }
            width: sc.borderAccent                 // 3
            radius: width / 2
            color: pal.ACC
        }

        Column {
            id: col
            width: parent.width

            // ── header row ───────────────────────────────────────────────
            Item {
                width: parent.width
                height: 42

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: sc.sp5      // 10
                    anchors.rightMargin: sc.sp5
                    spacing: sc.sp4                  // 8

                    Rectangle {                      // tinted icon-chip
                        visible: root.icon !== ""
                        Layout.preferredWidth: 22
                        Layout.preferredHeight: 22
                        radius: sc.radiusLg          // 6
                        color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.14)   // ACC @ 14%
                        Icon { anchors.centerIn: parent; name: root.icon
                               color: pal.ACC; size: 13 }
                    }
                    Text {
                        text: root.title; color: pal.TXT
                        font.pixelSize: sc.textMd; font.weight: Font.DemiBold
                        Layout.fillWidth: true
                        Layout.preferredWidth: 0
                        elide: Text.ElideRight
                    }
                    Icon {                               // one glyph that rotates
                        name: "chevron-down"
                        color: pal.TXT_MUTED; size: 14
                        rotation: root.expanded ? 0 : -90
                        Behavior on rotation { NumberAnimation { duration: Theme.reducedMotion ? 0 : 90; easing.type: Easing.OutCubic } }
                    }
                }

                Rectangle {                          // divider under open header
                    visible: root.expanded
                    anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
                    anchors.leftMargin: sc.borderAccent
                    height: 1; color: pal.BORDER
                }

                HoverHandler { id: hh; cursorShape: Qt.PointingHandCursor }
                TapHandler { onTapped: root.expanded = !root.expanded }
            }

            // ── animated content ─────────────────────────────────────────
            Item {
                width: parent.width
                clip: true
                height: root.expanded ? body.implicitHeight + sc.sp5 + sc.sp6 : 0
                Behavior on height {
                    NumberAnimation { duration: Theme.reducedMotion ? 0 : 160; easing.type: Easing.OutCubic }
                }
                Column {
                    id: body
                    x: sc.sp6; y: sc.sp5             // 12 / 10
                    width: parent.width - sc.sp6 * 2
                    spacing: sc.sp5                   // 10 between fields
                    // fade the fields as the section unfolds (lags the height)
                    opacity: root.expanded ? 1 : 0
                    Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : 120 } }
                }
            }
        }
    }
}
