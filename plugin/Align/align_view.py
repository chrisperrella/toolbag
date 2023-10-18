__author__ = "Chris Perrella"
__email__ = "perrella.chris@gmail.com"
__date__ = "10.17.2023"

import mset, os
import align_model

class AlignPlugin:
    def __init__( self ) -> None:
        self.window = mset.UIWindow( "Align Tools" )        
        self.window.width = 175

        self._render()
        self.window.visible = True

    def _render( self ):
        self.tool_logo_button = mset.UIButton()
        self.tool_logo_button.setIcon( self._get_icon_from_plugin_folder('tool.png') )
        self.tool_logo_button.frameless = True
        self.tool_logo_button.lit = False

        self.axis_align_label = mset.UILabel("Align to First Selection Axis:")
        self.axis_align_x = mset.UIButton()
        self.axis_align_x.setIcon(self._get_icon_from_plugin_folder('x.png'))
        self.axis_align_y = mset.UIButton()
        self.axis_align_y.setIcon(self._get_icon_from_plugin_folder('y.png'))
        self.axis_align_z = mset.UIButton()
        self.axis_align_z.setIcon(self._get_icon_from_plugin_folder('z.png'))

        self.axis_align_min_label = mset.UILabel("Align to Axis Min:")
        self.axis_align_min_x = mset.UIButton()
        self.axis_align_min_x.setIcon(self._get_icon_from_plugin_folder('x.png'))
        self.axis_align_min_y = mset.UIButton()
        self.axis_align_min_y.setIcon(self._get_icon_from_plugin_folder('y.png'))
        self.axis_align_min_z = mset.UIButton()
        self.axis_align_min_z.setIcon(self._get_icon_from_plugin_folder('z.png'))

        self.axis_align_max_label = mset.UILabel("Align to Axis Max:")
        self.axis_align_max_x = mset.UIButton()
        self.axis_align_max_x.setIcon(self._get_icon_from_plugin_folder('x.png'))
        self.axis_align_max_y = mset.UIButton()
        self.axis_align_max_y.setIcon(self._get_icon_from_plugin_folder('y.png'))
        self.axis_align_max_z = mset.UIButton()
        self.axis_align_max_z.setIcon(self._get_icon_from_plugin_folder('z.png'))

        self.axis_distribute_label = mset.UILabel("Distribute Along Axis:")
        self.axis_distribute_x = mset.UIButton()
        self.axis_distribute_x.setIcon(self._get_icon_from_plugin_folder('x.png'))
        self.axis_distribute_y = mset.UIButton()
        self.axis_distribute_y.setIcon(self._get_icon_from_plugin_folder('y.png'))
        self.axis_distribute_z = mset.UIButton()
        self.axis_distribute_z.setIcon(self._get_icon_from_plugin_folder('z.png'))
        self.axis_distribute_spacing = mset.UISliderFloat( min=0, max=250, name="Spacing" )

        self.axis_align_x.onClick = lambda: align_model.align( axis=0 )
        self.axis_align_y.onClick = lambda: align_model.align( axis=1 )
        self.axis_align_z.onClick = lambda: align_model.align( axis=2 )

        self.axis_align_min_x.onClick = lambda: align_model.align_to_min(0)
        self.axis_align_min_y.onClick = lambda: align_model.align_to_min(1)
        self.axis_align_min_z.onClick = lambda: align_model.align_to_min(2)

        self.axis_align_max_x.onClick = lambda: align_model.align_to_max(0)
        self.axis_align_max_y.onClick = lambda: align_model.align_to_max(1)
        self.axis_align_max_z.onClick = lambda: align_model.align_to_max(2)

        self.axis_distribute_x.onClick = lambda: align_model.distribute( axis=0, spacing=self.axis_distribute_spacing.value )
        self.axis_distribute_y.onClick = lambda: align_model.distribute( axis=1, spacing=self.axis_distribute_spacing.value )
        self.axis_distribute_z.onClick = lambda: align_model.distribute( axis=2, spacing=self.axis_distribute_spacing.value )

        #self.window.addElement(self.tool_logo_button)
        #self.window.addReturn()
        self.window.addElement(self.axis_align_label)
        self.window.addReturn()
        self.window.addElement(self.axis_align_x)
        self.window.addElement(self.axis_align_y)
        self.window.addElement(self.axis_align_z)
        self.window.addReturn() 
        self.window.addReturn()
        self.window.addElement(self.axis_align_min_label)
        self.window.addReturn()
        self.window.addElement(self.axis_align_min_x)
        self.window.addElement(self.axis_align_min_y)
        self.window.addElement(self.axis_align_min_z)
        self.window.addReturn()
        self.window.addReturn()
        self.window.addElement(self.axis_align_max_label)
        self.window.addReturn()
        self.window.addElement(self.axis_align_max_x)
        self.window.addElement(self.axis_align_max_y)
        self.window.addElement(self.axis_align_max_z)
        self.window.addReturn()
        self.window.addElement(self.axis_distribute_label)
        self.window.addReturn()
        self.window.addElement(self.axis_distribute_x)
        self.window.addElement(self.axis_distribute_y)
        self.window.addElement(self.axis_distribute_z)
        self.window.addReturn()
        self.window.addReturn()
        self.window.addElement(self.axis_distribute_spacing)

    def _get_icon_from_gui_folder(self, icon_name):
        path = os.path.join( os.getcwd(), 'data', 'gui', 'control', icon_name )
        if os.path.exists(path):
            return path
        else:
            print(f"[Align Tool]: Icon {icon_name} not found.")
            return None
        
    def _get_icon_from_plugin_folder(self, icon_name):
        path = os.path.join( os.getcwd(), 'data', 'plugin', 'Align', 'gui', icon_name )
        return path

    def _shutdown(self):
        mset.shutdownPlugin()

if __name__ == '__main__':
    AlignPlugin()
