#
# Azure theme
#

namespace eval ttk::theme::azure {
    variable colors
    array set colors {
        -accent         #0078d4
        -bg             #ffffff
        -fg             #000000
        -border         #d6d6d6
        -button         #ffffff
        -button-fg      #000000
        -select-bg      #cce8ff
        -select-fg      #000000
        -disabled-fg    #8e8e8e
        -active-bg      #f2f2f2
        -active-fg      #000000
        -focus          #0078d4
        -hover          #f2f2f2
        -hover-fg       #000000
        -entry-bg       #ffffff
        -entry-fg       #000000
        -entry-border   #d6d6d6
        -entry-focus    #0078d4
        -entry-select   #cce8ff
        -entry-disabled #f2f2f2
        -entry-readonly #f2f2f2
        -spin-button    #ffffff
        -spin-active    #f2f2f2
        -spin-disabled  #f2f2f2
        -progress-bg    #e6e6e6
        -progress-fg    #0078d4
        -tree-bg        #ffffff
        -tree-fg        #000000
        -tree-select    #cce8ff
        -tree-select-fg #000000
        -tree-heading   #f2f2f2
        -tree-heading-fg #000000
    }

    proc LoadImages {imgdir} {
        variable I
        foreach file [glob -nocomplain -directory $imgdir *.png] {
            set name [file rootname [file tail $file]]
            if {[info exists I($name)]} continue
            set I($name) [image create photo -file $file]
        }
    }

    proc ReloadImages {imgdir} {
        variable I
        foreach name [array names I] {
            $I($name) configure -file [file join $imgdir $name.png]
        }
    }

    proc StyleTButton {colors} {
        ttk::style element create Button.button vsapi BUTTON 1 {pressed !disabled} \
            -padding {12 5 12 5} -border 4
        ttk::style layout TButton {
            Button.button -children {
                Button.focus -children {
                    Button.padding -children {
                        Button.label
                    }
                }
            }
        }
        ttk::style configure TButton \
            -background $colors(-button) \
            -foreground $colors(-fg) \
            -bordercolor $colors(-border) \
            -focuscolor $colors(-focus) \
            -lightcolor $colors(-button) \
            -darkcolor $colors(-button) \
            -anchor center \
            -width -12
        ttk::style map TButton \
            -background [list disabled $colors(-button) \
                active $colors(-active-bg) \
                pressed $colors(-active-bg)] \
            -foreground [list disabled $colors(-disabled-fg)] \
            -bordercolor [list disabled $colors(-border) \
                pressed $colors(-border) \
                focus $colors(-focus) \
                hover $colors(-focus)]
    }

    proc StyleTCheckbutton {colors} {
        ttk::style element create Checkbutton.indicator vsapi CHECKBUTTON 1 {pressed !disabled} \
            -padding 0 -border 0
        ttk::style layout TCheckbutton {
            Checkbutton.padding -children {
                Checkbutton.indicator -side left
                Checkbutton.focus -children {
                    Checkbutton.label -side right
                }
            }
        }
        ttk::style configure TCheckbutton \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -indicatorbackground $colors(-entry-bg) \
            -indicatorforeground $colors(-fg) \
            -indicatordisabled $colors(-disabled-fg) \
            -indicatorreadonly $colors(-entry-readonly) \
            -indicatormargin {0 0 4 0}
        ttk::style map TCheckbutton \
            -indicatorbackground [list readonly $colors(-entry-readonly) \
                disabled $colors(-entry-readonly) \
                selected $colors(-accent) \
                hover $colors(-entry-bg) \
                pressed $colors(-accent) \
                active $colors(-accent)] \
            -indicatorforeground [list readonly $colors(-fg) \
                disabled $colors(-disabled-fg) \
                selected $colors(-bg) \
                pressed $colors(-bg) \
                active $colors(-bg)]
    }

    proc StyleTRadiobutton {colors} {
        ttk::style element create Radiobutton.indicator vsapi RADIOBUTTON 1 {pressed !disabled} \
            -padding 0 -border 0
        ttk::style layout TRadiobutton {
            Radiobutton.padding -children {
                Radiobutton.indicator -side left
                Radiobutton.focus -children {
                    Radiobutton.label -side right
                }
            }
        }
        ttk::style configure TRadiobutton \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -indicatorbackground $colors(-entry-bg) \
            -indicatorforeground $colors(-fg) \
            -indicatordisabled $colors(-disabled-fg) \
            -indicatorreadonly $colors(-entry-readonly) \
            -indicatormargin {0 0 4 0}
        ttk::style map TRadiobutton \
            -indicatorbackground [list readonly $colors(-entry-readonly) \
                disabled $colors(-entry-readonly) \
                selected $colors(-accent) \
                hover $colors(-entry-bg) \
                pressed $colors(-accent) \
                active $colors(-accent)] \
            -indicatorforeground [list readonly $colors(-fg) \
                disabled $colors(-disabled-fg) \
                selected $colors(-bg) \
                pressed $colors(-bg) \
                active $colors(-bg)]
    }

    proc StyleTCombobox {colors} {
        ttk::style element create Combobox.field vsapi ENTRY 1 {focus !disabled} \
            -border 4 -padding {12 5 12 5}
        ttk::style layout TCombobox {
            Combobox.field -side left -expand 1 -children {
                Combobox.padding -children {
                    Combobox.textarea
                }
            }
            Combobox.downarrow -side right
        }
        ttk::style configure TCombobox \
            -background $colors(-entry-bg) \
            -foreground $colors(-fg) \
            -bordercolor $colors(-border) \
            -focuscolor $colors(-focus) \
            -lightcolor $colors(-entry-bg) \
            -darkcolor $colors(-entry-bg) \
            -arrowcolor $colors(-fg) \
            -arrowsize 12
        ttk::style map TCombobox \
            -background [list readonly $colors(-entry-readonly) \
                disabled $colors(-entry-disabled)] \
            -foreground [list disabled $colors(-disabled-fg)] \
            -bordercolor [list readonly $colors(-border) \
                disabled $colors(-border) \
                focus $colors(-focus) \
                hover $colors(-focus)] \
            -arrowcolor [list disabled $colors(-disabled-fg)]
    }

    proc StyleTEntry {colors} {
        ttk::style element create Entry.field vsapi ENTRY 1 {focus !disabled} \
            -border 4 -padding {12 5 12 5}
        ttk::style layout TEntry {
            Entry.field -children {
                Entry.padding -children {
                    Entry.textarea
                }
            }
        }
        ttk::style configure TEntry \
            -background $colors(-entry-bg) \
            -foreground $colors(-fg) \
            -bordercolor $colors(-border) \
            -focuscolor $colors(-focus) \
            -lightcolor $colors(-entry-bg) \
            -darkcolor $colors(-entry-bg) \
            -insertcolor $colors(-fg)
        ttk::style map TEntry \
            -background [list readonly $colors(-entry-readonly) \
                disabled $colors(-entry-disabled)] \
            -foreground [list disabled $colors(-disabled-fg)] \
            -bordercolor [list readonly $colors(-border) \
                disabled $colors(-border) \
                focus $colors(-focus) \
                hover $colors(-focus)]
    }

    proc StyleTFrame {colors} {
        ttk::style configure TFrame \
            -background $colors(-bg)
    }

    proc StyleTLabel {colors} {
        ttk::style configure TLabel \
            -background $colors(-bg) \
            -foreground $colors(-fg)
    }

    proc StyleTNotebook {colors} {
        ttk::style element create Notebook.client vsapi FRAME 0
        ttk::style layout TNotebook.Tab {
            Notebook.tab -children {
                Notebook.padding -children {
                    Notebook.focus -children {
                        Notebook.label
                    }
                }
            }
        }
        ttk::style configure TNotebook \
            -background $colors(-bg) \
            -bordercolor $colors(-border)
        ttk::style configure TNotebook.Tab \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -padding {6 6 6 6}
        ttk::style map TNotebook.Tab \
            -background [list selected $colors(-bg) \
                !selected $colors(-bg) \
                hover $colors(-hover)] \
            -foreground [list selected $colors(-accent)] \
            -expand [list selected {0 0 0 2}]
    }

    proc StyleTProgressbar {colors} {
        ttk::style element create Progressbar.trough vsapi PROGRESS 1
        ttk::style element create Progressbar.pbar vsapi PROGRESS 2
        ttk::style layout TProgressbar {
            Progressbar.trough -children {
                Progressbar.pbar
            }
        }
        ttk::style configure TProgressbar \
            -background $colors(-progress-fg) \
            -troughcolor $colors(-progress-bg) \
            -bordercolor $colors(-border)
    }

    proc StyleTScale {colors} {
        ttk::style element create Scale.trough vsapi TRACKBAR 1
        ttk::style element create Scale.slider vsapi THUMB 1 {pressed !disabled}
        ttk::style layout TScale {
            Scale.focus -children {
                Scale.trough -side top -expand 1
                Scale.slider -side top
            }
        }
        ttk::style configure TScale \
            -background $colors(-bg) \
            -troughcolor $colors(-progress-bg) \
            -sliderbackground $colors(-button) \
            -sliderforeground $colors(-fg) \
            -sliderlength 30
        ttk::style map TScale \
            -sliderbackground [list active $colors(-active-bg) \
                pressed $colors(-active-bg)]
    }

    proc StyleTScrollbar {colors} {
        ttk::style element create Scrollbar.trough vsapi SCROLLBAR 1
        ttk::style element create Scrollbar.thumb vsapi SCROLLBAR 2 {pressed !disabled}
        ttk::style layout TScrollbar {
            Scrollbar.trough -children {
                Scrollbar.thumb
            }
        }
        ttk::style configure TScrollbar \
            -background $colors(-button) \
            -troughcolor $colors(-progress-bg) \
            -bordercolor $colors(-border) \
            -arrowcolor $colors(-fg) \
            -arrowsize 16
        ttk::style map TScrollbar \
            -background [list active $colors(-active-bg) \
                pressed $colors(-active-bg)]
    }

    proc StyleTSeparator {colors} {
        ttk::style element create Separator.separator vsapi FRAME 0
        ttk::style layout TSeparator {
            Separator.separator
        }
        ttk::style configure TSeparator \
            -background $colors(-border)
    }

    proc StyleTSizegrip {colors} {
        ttk::style element create Sizegrip.sizegrip vsapi GRIP 1
        ttk::style layout TSizegrip {
            Sizegrip.sizegrip -side bottom -sticky se
        }
        ttk::style configure TSizegrip \
            -background $colors(-bg)
    }

    proc StyleTSpinbox {colors} {
        ttk::style element create Spinbox.field vsapi ENTRY 1 {focus !disabled} \
            -border 4 -padding {12 5 12 5}
        ttk::style layout TSpinbox {
            Spinbox.field -side left -expand 1 -children {
                Spinbox.padding -children {
                    Spinbox.textarea
                }
            }
            Spinbox.buttons -side right -children {
                Spinbox.uparrow -side top
                Spinbox.downarrow -side bottom
            }
        }
        ttk::style configure TSpinbox \
            -background $colors(-entry-bg) \
            -foreground $colors(-fg) \
            -bordercolor $colors(-border) \
            -focuscolor $colors(-focus) \
            -lightcolor $colors(-entry-bg) \
            -darkcolor $colors(-entry-bg) \
            -insertcolor $colors(-fg) \
            -arrowcolor $colors(-fg) \
            -arrowsize 10
        ttk::style map TSpinbox \
            -background [list readonly $colors(-entry-readonly) \
                disabled $colors(-entry-disabled)] \
            -foreground [list disabled $colors(-disabled-fg)] \
            -bordercolor [list readonly $colors(-border) \
                disabled $colors(-border) \
                focus $colors(-focus) \
                hover $colors(-focus)] \
            -arrowcolor [list disabled $colors(-disabled-fg)]
    }

    proc StyleTreeview {colors} {
        ttk::style element create Treeview.field vsapi FRAME 0
        ttk::style layout Treeview {
            Treeview.field -children {
                Treeview.padding -children {
                    Treeview.treearea
                }
            }
        }
        ttk::style configure Treeview \
            -background $colors(-tree-bg) \
            -foreground $colors(-tree-fg) \
            -fieldbackground $colors(-tree-bg)
        ttk::style map Treeview \
            -background [list selected $colors(-tree-select)] \
            -foreground [list selected $colors(-tree-select-fg)]
        ttk::style configure Treeview.Heading \
            -background $colors(-tree-heading) \
            -foreground $colors(-tree-heading-fg) \
            -font {TkDefaultFont 9 bold}
    }

    proc SetStyle {light} {
        variable colors
        if {$light} {
            array set colors {
                -accent         #0078d4
                -bg             #ffffff
                -fg             #000000
                -border         #d6d6d6
                -button         #ffffff
                -button-fg      #000000
                -select-bg      #cce8ff
                -select-fg      #000000
                -disabled-fg    #8e8e8e
                -active-bg      #f2f2f2
                -active-fg      #000000
                -focus          #0078d4
                -hover          #f2f2f2
                -hover-fg       #000000
                -entry-bg       #ffffff
                -entry-fg       #000000
                -entry-border   #d6d6d6
                -entry-focus    #0078d4
                -entry-select   #cce8ff
                -entry-disabled #f2f2f2
                -entry-readonly #f2f2f2
                -spin-button    #ffffff
                -spin-active    #f2f2f2
                -spin-disabled  #f2f2f2
                -progress-bg    #e6e6e6
                -progress-fg    #0078d4
                -tree-bg        #ffffff
                -tree-fg        #000000
                -tree-select    #cce8ff
                -tree-select-fg #000000
                -tree-heading   #f2f2f2
                -tree-heading-fg #000000
            }
        } else {
            array set colors {
                -accent         #0078d4
                -bg             #2d2d2d
                -fg             #ffffff
                -border         #4e4e4e
                -button         #3c3c3c
                -button-fg      #ffffff
                -select-bg      #005a9e
                -select-fg      #ffffff
                -disabled-fg    #8e8e8e
                -active-bg      #4e4e4e
                -active-fg      #ffffff
                -focus          #0078d4
                -hover          #4e4e4e
                -hover-fg       #ffffff
                -entry-bg       #3c3c3c
                -entry-fg       #ffffff
                -entry-border   #4e4e4e
                -entry-focus    #0078d4
                -entry-select   #005a9e
                -entry-disabled #2d2d2d
                -entry-readonly #2d2d2d
                -spin-button    #3c3c3c
                -spin-active    #4e4e4e
                -spin-disabled  #2d2d2d
                -progress-bg    #4e4e4e
                -progress-fg    #0078d4
                -tree-bg        #2d2d2d
                -tree-fg        #ffffff
                -tree-select    #005a9e
                -tree-select-fg #ffffff
                -tree-heading   #3c3c3c
                -tree-heading-fg #ffffff
            }
        }

        StyleTButton $colors
        StyleTCheckbutton $colors
        StyleTRadiobutton $colors
        StyleTCombobox $colors
        StyleTEntry $colors
        StyleTFrame $colors
        StyleTLabel $colors
        StyleTNotebook $colors
        StyleTProgressbar $colors
        StyleTScale $colors
        StyleTScrollbar $colors
        StyleTSeparator $colors
        StyleTSizegrip $colors
        StyleTSpinbox $colors
        StyleTreeview $colors
    }
}

ttk::style theme create azure -parent clam -settings {
    ttk::theme::azure::SetStyle 1
}

ttk::style theme create azure-dark -parent clam -settings {
    ttk::theme::azure::SetStyle 0
}
