import os
import numpy as np
from PyQt5 import QtWidgets, uic
import matplotlib.pyplot as plt
from tqdm import tqdm

from ceyehao.utils.utils import Timer
from ceyehao.tools.infer import TTPredictor
from ceyehao.utils.data_process import (
    gen_pin_tensor,
    p_transform,
    tt_convert,
    obs_param_convert,
)
from ceyehao.utils.visualization import (
    plot_tt_vf,
    plot_obstacle,
    plot_fp_tensor,
    fig2np,
    generate_autocad_script,
    create_obstacle_figure,
)
from ceyehao.utils.io import (
    read_obs_param,
    write_obs_param,
    read_obs_param_list,
    write_obs_param_list,
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        print("Initializing the app...")
        # Try to find the UI file in different possible locations
        ui_paths = [
            "ceyehao/gui/mainwindow.ui",
            "gui/mainwindow.ui",
            os.path.join(os.path.dirname(__file__), "mainwindow.ui")
        ]
        
        ui_loaded = False
        for ui_path in ui_paths:
            if os.path.exists(ui_path):
                uic.loadUi(ui_path, self)
                ui_loaded = True
                break
        
        if not ui_loaded:
            raise FileNotFoundError(f"Could not find mainwindow.ui in any of: {ui_paths}")
        # set window title
        self.setWindowTitle(
            "FlowFormer: Microfluidic Channel Design For Custom Flow Shapes"
        )
        self.cfg = cfg
        # channel metadata
        self.res = self.cfg.profile_size  # profile resolution
        print(f"Profile resolution: {self.res}")

        # initialize the predictor
        self.predictor = TTPredictor(cfg=self.cfg)
        self.predictor.profile_size = self.res
        # bands parameters given in percentage values.
        self.chn_dim = [300, 300]  # [width, height] in microns
        self.num_obs = 8
        # obstacle image aspect ratio 300 by 120
        self.sb_0y.setValue(0)
        self.sb_5y.setValue(self.chn_dim[1])

        # Initialize variables to save the final parameters
        # input profile color band params
        self.pin_param_fin = np.zeros(6, dtype=np.int16)

        # FINAL obstacle coords and wall-position
        self.obs_coord_fin = np.zeros((self.num_obs, 18))  # i~[1,5] [xi0,xi1,yi]
        self.opm = {True: "Top", False: "Bottom"}  # obstacle position Map
        self.obs_pos_fin = np.array([False for i in range(self.num_obs)])  # True for Top Wall.

        # the profile imgs, from input image to output image, RGB float 0~1
        self.profiles_fin = np.zeros(
            (self.num_obs + 1, self.res[0], self.res[1], 3), dtype=np.uint8
        )
        self.tt_fin = np.zeros(
            (self.num_obs, self.res[0], self.res[1], 2), dtype=np.int16
        )

        # initialize variables for storing temp designs and graphs, and background processes
        self.pin_param_tmp = np.zeros(6, dtype=np.int16)  # for writing to the spinbox
        # for drawings. / avoid conflict with the upadte slot
        self.pin_param_tmp = np.zeros(6, dtype=np.int16)
        self.obs_coord_tmp = np.zeros(18)
        self.obs_pos_tmp = False

        self.pint_tmp = np.zeros((self.res[0], self.res[1], 3))
        self.tt_tmp = np.zeros((self.res[0], self.res[1], 2), dtype=np.int16)

        # background image for feeding into the prediction algorithm
        self.bkgd_obs_fig, self.bkgd_obs_axe = create_obstacle_figure()
        # numpy array of obstacle img converted from the matplotlib figure. for prediction input
        self.bkgd_obs_mat = None

        self.direct_result = False
        # prevent iteratively trigger the signals and slots during updating the values of widgets
        self.flag_getting = False
        self.selected_obstacle = 1

        # wrappers for the widgets in UI.
        self.obstacle_canvas = {
            "1": self.step1,
            "2": self.step2,
            "3": self.step3,
            "4": self.step4,
            "5": self.step5,
            "6": self.step6,
            "7": self.step7,
            "8": self.step8,
        }

        self.profiles_canvas = {
            "0": self.p_in,
            "1": self.p_int1,
            "2": self.p_int2,
            "3": self.p_int3,
            "4": self.p_int4,
            "5": self.p_int5,
            "6": self.p_int6,
            "7": self.p_int7,
            "8": self.p_int8,
            "-1": self.pint_preview,
        }

        self.obs_sbs = [
            self.sb_0x0,
            self.sb_0x1,
            self.sb_0y,
            self.sb_1x0,
            self.sb_1x1,
            self.sb_1y,
            self.sb_2x0,
            self.sb_2x1,
            self.sb_2y,
            self.sb_3x0,
            self.sb_3x1,
            self.sb_3y,
            self.sb_4x0,
            self.sb_4x1,
            self.sb_4y,
            self.sb_5x0,
            self.sb_5x1,
            self.sb_5y,
        ]

        self.pin_sbs = [
            self.sb_1l,
            self.sb_1r,
            self.sb_2l,
            self.sb_2r,
            self.sb_3l,
            self.sb_3r,
        ]

        # signal/slots for obstacle settings
        self.sselect.currentTextChanged.connect(self.update_selected_obs)
        self.pb_sget.clicked.connect(self.get_obs_param)
        self.pb_sset.clicked.connect(self.set_obs_param)
        self.tb_switch.clicked.connect(self.tb_switch_clicked)
        for item in self.obs_sbs:
            item.valueChanged.connect(self.update_obs_tmp)

        # signal/slots for input profile settings
        self.pb_pinget.clicked.connect(self.get_pin_param)
        self.pb_pinset.clicked.connect(self.set_pin_param)
        for item in self.pin_sbs:
            item.valueChanged.connect(self.update_pin_preview)

        # other signal/slots
        self.pb_auto.clicked.connect(self.auto_switch)
        self.pb_compute.clicked.connect(self.compute)
        self.pb_init.clicked.connect(self.init_obs)
        self.pb_clr.clicked.connect(self.clear_obs)
        self.pb_mirr.clicked.connect(self.obs_coord_mirror)

        # file menu signal/slots
        self.actLStep.triggered.connect(self.load_obs)
        self.actLSeq.triggered.connect(self.load_sequence)
        self.actSStep.triggered.connect(self.save_obs)
        self.actSSeq.triggered.connect(self.save_sequence)
        self.actSP.triggered.connect(self.save_profiles)
        self.actGCAD.triggered.connect(self.save_CAD_script)
        self.actExpTtPlot.triggered.connect(self.export_TT_plot)
        self.actExpObsPlot.triggered.connect(self.export_Obs_plot)

        print("App initialized.")

    ########################
    #  App Core Functions  #
    ########################

    def update_obs_tmp(self):
        """update the temp obstacle coord and the preview canvas to match the spinbox displays"""

        if not self.flag_getting:
            for i in range(len(self.obs_sbs)):
                self.obs_coord_tmp[i] = self.obs_sbs[i].value() / 1e6

            self.draw_obs(self.obs_coord_tmp, self.obs_pos_tmp, self.s_preview)
            if self.direct_result:  # directly update the finalized obstacle parameters
                self.auto_update()

    def tb_switch_clicked(self, checked):
        self.obs_pos_tmp = checked
        self.tb_switch.setText(self.opm[checked])
        self.draw_obs(self.obs_coord_tmp, self.obs_pos_tmp, self.s_preview)
        if self.direct_result:
            self.auto_update()

    def _refresh_obs_param_panel(self):
        self.flag_getting = True
        for i, item in enumerate(self.obs_sbs):
            item.setValue(self.obs_coord_tmp[i] * 1e6)
        self.flag_getting = False

        self.tb_switch.setChecked(self.obs_pos_tmp)
        self.tb_switch.setText(self.opm[self.obs_pos_tmp])

    def get_obs_param(self):
        """get the parameters of the selected obstacle in final list."""
        # update the tmp obstacle coord, pos and the tt from fin obstacle list
        self.obs_coord_tmp = self.obs_coord_fin[self.selected_obstacle - 1].copy()
        self.obs_pos_tmp = self.obs_pos_fin[self.selected_obstacle - 1].copy()
        self.tt_tmp[:] = self.tt_fin[self.selected_obstacle - 1].copy()

        self._refresh_obs_param_panel()
        self.draw_obs(self.obs_coord_tmp, self.obs_pos_tmp, self.s_preview)
        self.draw_tt(self.tt_tmp, self.dtm_preview)
        self.draw_profile(self.profiles_fin[self.selected_obstacle], self.pint_preview)

    def set_obs_param(self):
        self.obs_coord_fin[self.selected_obstacle - 1] = self.obs_coord_tmp.copy()
        self.obs_pos_fin[self.selected_obstacle - 1] = self.obs_pos_tmp

        print(
            f"Obsacle #{self.selected_obstacle} param updated:\n{self.obs_coord_fin[self.selected_obstacle-1]}"
        )
        print(
            f'Pos: {("Top" if self.obs_pos_fin[self.selected_obstacle-1] else "Bottom")}\n'
        )
        print("------------------")

        self.draw_obs(
            self.obs_coord_fin[self.selected_obstacle - 1],
            self.obs_pos_fin[self.selected_obstacle - 1],
            self.obstacle_canvas[str(self.selected_obstacle)],
        )

        self.update_tt_p_temp()
        self.tt_fin[self.selected_obstacle - 1] = self.tt_tmp.copy()
        self.draw_tt(self.tt_tmp, self.dtm_preview)
        self.update_pint_fin(self.selected_obstacle)

    def update_pin_preview(self):
        if self.flag_getting == False:
            for i, item in enumerate(self.pin_sbs):
                self.pin_param_tmp[i] = item.value()
            self.draw_pin(self.pin_param_tmp, self.pin_preview)
            if self.direct_result == True:
                self.auto_update(True)

    def get_pin_param(self):
        self.flag_getting = True
        self.pin_param_tmp[:] = self.pin_param_fin.copy()
        for i, item in enumerate(self.pin_sbs):
            item.setValue(self.pin_param_tmp[i])
        self.flag_getting = False

        self.draw_pin(self.pin_param_tmp, self.pin_preview)

    def set_pin_param(self):
        self.pin_param_fin[:] = self.pin_param_tmp.copy()
        print(f"Input profile param updated:\n{self.pin_param_fin}")
        self.profiles_fin[0] = self.draw_pin(self.pin_param_fin, self.p_in)
        self.update_pint_fin(1)

    def update_tt_p_temp(self):
        """Prepare the obstacle image for neural network input."""
        print("Updating temporary tt and profile...")
        # 1. from the temp obstacle param, plot a normal obstacle image as the input of predict_tm
        plot_obstacle(self.obs_coord_tmp, False, self.bkgd_obs_axe)
        self.bkgd_obs_fig.canvas.draw()
        self.bkgd_obs_mat = fig2np(self.bkgd_obs_fig)
        # 2. feed the obstacle image into the prediction method to get the tt.
        with Timer() as t:
            self.tt_tmp = self.predict_tt(self.bkgd_obs_mat, self.obs_pos_tmp)
        self.draw_tt(self.tt_tmp, self.dtm_preview)
        # self.tt_fin[self.selected_obstacle - 1] = self.tt_tmp???
        # update the pint preview
        self.pint_tmp = self.compute_p(
            self.profiles_fin[self.selected_obstacle - 1], self.tt_tmp
        )
        self.draw_profile(self.pint_tmp, self.pint_preview)
        print(f"Temporary tt and profile updated.")

    def update_tt_fin(self, obs_no):
        """update the tt_fin of the selected obstacle"""
        print(f"Updating tt of obstacle {obs_no}")
        # 1. from the temp obstacle param, plot a normal obstacle image as the input of predict_tt
        plot_obstacle(self.obs_coord_fin[obs_no - 1], False, self.bkgd_obs_axe)
        self.bkgd_obs_fig.canvas.draw()
        self.bkgd_obs_mat = fig2np(self.bkgd_obs_fig)
        # 2. feed the obstacle image into the prediction method to get the tt.
        with Timer() as t:
            self.tt_tmp = self.predict_tt(
                self.bkgd_obs_mat, self.obs_pos_fin[obs_no - 1]
            )
        self.tt_fin[obs_no - 1] = self.tt_tmp.copy()
        print(f"tt of obstacle {obs_no} updated")

    def update_pint_fin(self, start_obs_no):
        """update all the intermediate and final profiles.

        Args:
            start_obstacle (int): the obstacle number (start from 1) that is changed. set 0 for pin change.
        """
        for i in range(start_obs_no, 9):
            self.profiles_fin[i] = self.compute_p(
                self.profiles_fin[i - 1], self.tt_fin[i - 1]
            )
            self.draw_profile(self.profiles_fin[i], self.profiles_canvas[str(i)])
        self.draw_profile(self.profiles_fin[8], self.p_out)

    #############################################
    #  Computation and Visualizatoin Functions  #
    #############################################

    def draw_obs(self, param, pos, widget):
        """plot the obstacle, plus the upper and lower bound of the channel on to the canvas widget."""
        xlim0 = param[0::3].min()
        xlim1 = param[1::3].max()
        plot_obstacle(param, pos, widget.ax, for_infer=False)
        widget.ax.plot([xlim0, xlim1], [0, 0], "k")
        widget.ax.plot([xlim0, xlim1], [param[17], param[17]], "k")
        widget.draw()
        widget.ax.set_ylim(0, param[17])

    def draw_pin(self, param, widget):
        pin_mat = gen_pin_tensor(param.reshape(3,2).T, self.res, param_scale=True).squeeze()
        plot_fp_tensor(pin_mat, widget.ax)
        widget.draw()
        return pin_mat

    def draw_tt(self, tt, widget):
        plot_tt_vf(tt, widget.ax)
        widget.draw()

    def draw_profile(self, profile_img, widget):
        plot_fp_tensor(profile_img, widget.ax)
        widget.draw()

    def predict_tt(self, obs_img, pos):
        tt_hat = self.predictor.predict_from_obs_img(obs_img)
        if pos:
            tt_hat = tt_convert(tt_hat, vert_sym=True)

        return tt_hat

    def compute_p(self, pin, tt):
        pout = p_transform(pin, tt)
        return pout

    ############################
    # App Auxiliary Functions  #
    ############################

    def update_selected_obs(self, x):
        self.selected_obstacle = int(x)

    def refresh_app(self):
        self.update()

    def auto_switch(self, checked):
        if checked == True:
            self.direct_result = True
            print("automatic mode on!")
        elif checked == False:
            self.direct_result = False
            print("automatic mode off!")

    def compute(self):
        self.update_tt_p_temp()
        self.draw_tt(self.tt_tmp, self.dtm_preview)

    def init_obs(self):
        self.flag_getting = True
        for item in self.obs_sbs[0::3]:
            item.setValue(0)
        for item in self.obs_sbs[1::3]:
            item.setValue(150)
        for i, item in enumerate(self.obs_sbs[2::3]):
            item.setValue(i * 60)

        self.obs_pos_tmp = False
        self.tb_switch.setText("Bottom")
        self.tb_switch.setChecked(False)
        self.flag_getting = False
        self.update_obs_tmp()

    def clear_obs(self):
        self.flag_getting = True
        for item in self.obs_sbs[:17]:
            item.setValue(0)
        self.obs_pos_tmp = False
        self.tb_switch.setText("Bottom")
        self.tb_switch.setChecked(False)
        self.flag_getting = False
        self.update_obs_tmp()

    def obs_coord_mirror(self):
        self.obs_coord_tmp = obs_param_convert(self.obs_coord_tmp)
        self.flag_getting = True
        for i, item in enumerate(self.obs_sbs):
            item.setValue(self.obs_coord_tmp[i] * 1e6)
        self.flag_getting = False
        self.update_obs_tmp()

    def auto_update(self, update_pin=False):
        """automatic assign temp inlet profile and obstacle to the finalized list, and update the tt list and temp profile"""
        if update_pin:  # when inlet profile is modified
            self.pin_param_fin[:] = self.pin_param_tmp.copy()
            self.profiles_fin[0] = self.draw_pin(self.pin_param_fin, self.p_in)
        else:  # or when the obstacle is modified
            self.obs_coord_fin[self.selected_obstacle - 1] = self.obs_coord_tmp.copy()
            self.obs_pos_fin[self.selected_obstacle - 1] = self.obs_pos_tmp
            print(
                f"obstacle #{self.selected_obstacle} param updated:\
                    \n{self.obs_coord_fin[self.selected_obstacle-1]}"
            )
            print(
                f'Pos: {("Top" if self.obs_pos_fin[self.selected_obstacle-1] else "Bottom")}\n'
            )
            print("------------------")

            # update temp previews and tt list
            self.update_tt_p_temp()
            self.tt_fin[self.selected_obstacle - 1] = self.tt_tmp.copy()
            # update final obstacle view.
            self.draw_obs(
                self.obs_coord_fin[self.selected_obstacle - 1],
                self.obs_pos_fin[self.selected_obstacle - 1],
                self.obstacle_canvas[str(self.selected_obstacle)],
            )
        # update all intermediate profiles.
        self.update_pint_fin(1 if update_pin else self.selected_obstacle)

    def load_obs(self):
        """load the coord and the pos to the temp params."""
        fs = FileSelector(self, "Load obstacle")
        try:
            if fs.exec_():
                fname = fs.selectedFiles()
            try:
                with open(fname[0], "r") as f:
                    obs_no, coords, pos = read_obs_param(f)
                self.obs_coord_tmp = coords
                self.obs_pos_tmp = pos
                print(f"Loaded obstacle from: {fname}")
                self.draw_obs(self.obs_coord_tmp, self.obs_pos_tmp, self.s_preview)
                self._refresh_obs_param_panel()
            except:
                print("Loading unscuccessful.")
        except:
            print("File not selected.")

    def load_sequence(self):
        """load the obstacles in the file to the indexed position"""
        fs = FileSelector(self, "Load Sequence")
        try:
            if fs.exec_():
                fname = fs.selectedFiles()

            try:
                with Timer() as t:
                    obs_no_list = []
                    with open(fname[0], "r") as f:
                        obs_no_list, coords, pos = read_obs_param_list(f)
                        self.obs_coord_fin[obs_no_list - 1] = coords
                        self.obs_pos_fin[obs_no_list - 1] = pos
                    for i in obs_no_list:
                        self.draw_obs(
                            self.obs_coord_fin[i - 1],
                            self.obs_pos_fin[i - 1],
                            self.obstacle_canvas[str(i)],
                        )
                        self.update_tt_fin(i)
                    self.update_pint_fin(min(obs_no_list))
                    # self.draw_obs(self.obs_coord_tmp, self.obs_pos_tmp, self.s_preview)
                    print(f"Loaded sequence from: {fname}")
                    print(f"loaded obstacles: {obs_no_list}")
            except ValueError:
                print("obstacle info is incorrect.")
            except KeyError:
                print(f"obstacle number is incorrect: {obs_no_list}")
            except:
                print("Loading unscuccessful for undetermined causes.")
        except:
            print("File selection cancelled.")

    def save_obs(self):
        """save the coord and the pos of the temp obstacle."""
        fs = FileSelector(self, "Save obstacle")
        try:
            if fs.exec_():
                fname = fs.selectedFiles()
            try:
                with open(fname[0], "w") as f:
                    write_obs_param(f, 0, self.obs_coord_tmp, self.obs_pos_tmp)
                print(f"write obstacle to: {fname}")
            except:
                print("Writing unsuccessful.")
        except:
            print("File not selected.")

    def save_sequence(self):
        """save the coord and the pos of the temp obstacle."""
        fs = FileSelector(self, "Save obstacle")

        try:
            text, ok = QtWidgets.QInputDialog().getText(
                self,
                "Save Sequence",
                "The list of the obstacles to be saved (separated by ','):",
                QtWidgets.QLineEdit.Normal,
                "1, 2, 3, 4, 5, 6, 7, 8",
            )
            try:
                assert ok
                obs_no_list = np.fromstring(text, dtype=np.int8, sep=",")
                print(f"obstacle no:{obs_no_list}")
                if fs.exec_():
                    fname = fs.selectedFiles()[0]
                    if not fname.endswith(".txt"):
                        fname += ".txt"
                try:
                    with open(fname, "w") as f:
                        write_obs_param_list(
                            f,
                            obs_no_list,
                            self.obs_coord_fin[obs_no_list - 1],
                            self.obs_pos_fin[obs_no_list - 1],
                        )
                    print(f"write sequence to: {fname}")
                except:
                    print("Writing unsuccessful")
            except:
                print("File not selected.")
        except:
            print("Wrong obstacle list!")

    def save_profiles(self):
        """save the profiles to the selected folder."""
        fs = FileSelector(self, "Folder")

        try:
            p_idx_string, ok1 = QtWidgets.QInputDialog().getText(
                self,
                "Save Sequence",
                "The list of the profiles to be saved (separated by ',')\n0 represent inlet profile, -1 represent temp preview profiles):",
                QtWidgets.QLineEdit.Normal,
                "0, 1, 2, 3, 4, 5, 6, 7, 8, -1",
            )
            if not ok1:
                print("user terminated save.")
                return
            profile_no_list = np.fromstring(p_idx_string, dtype=np.int8, sep=",")
            print(f"profiles no:{profile_no_list}")

            series_name, ok2 = QtWidgets.QInputDialog().getText(
                self,
                "series name",
                "series name of the profiles to be saved:",
                QtWidgets.QLineEdit.Normal,
                "pred",
            )
            if not ok2:
                print("user terminated save.")
                return

            if fs.exec_():
                save_path = fs.selectedFiles()[0]
            else:
                print("File not selected.")
                return

            try:
                for i in profile_no_list:
                    self.profiles_canvas[str(i)].fig.savefig(
                        os.path.join(save_path, f"{series_name}_p{i}.png"),
                        dpi=600,
                        bbox_inches="tight",
                        pad_inches=0,
                    )

                print(f"Profiles saved to: {save_path}")
            except:
                print("Writing profiles to files unsuccessful.")
        except:
            print("Save profile failed due to error.")

    def save_CAD_script(self):
        fs = FileSelector(self, "select script directory")
        try:
            if fs.exec_():
                fname = fs.selectedFiles()
            try:
                generate_autocad_script(fname[0], self.obs_coord_fin)
            except:
                print("Writing unsuccessful.")
        except:
            print("File not selected.")
        pass

    def export_TT_plot(self):
        """save the final transformation tensor plots to the selected folder."""
        fs = FileSelector(self, "Folder")

        try:
            tt_list_text, ok1 = QtWidgets.QInputDialog().getText(
                self,
                "Save TT plots",
                "The list of the transformation tensors to be plotted (separated by ','):",
                QtWidgets.QLineEdit.Normal,
                "1, 2, 3, 4, 5, 6, 7, 8",
            )
        except:
            print("Wrong obstacle list!")
        try:
            tt_plot_res_text, ok2 = QtWidgets.QInputDialog().getText(
                self,
                "Save TT plots",
                "Specify the plot resolution of transformation tensors:",
                QtWidgets.QLineEdit.Normal,
                "100",
            )
        except:
            print("Wrong obstacle list!")
        try:
            assert ok1 & ok2
            tt_no_list = np.fromstring(tt_list_text, dtype=np.int8, sep=",")
            tt_plot_res = [int(tt_plot_res_text)] * 2

            print(f"tts no:{tt_no_list}")
            if fs.exec_():
                save_path = fs.selectedFiles()[0]
        except:
            print("File not selected.")

        try:
            for i in tqdm(tt_no_list):
                fname = os.path.join(save_path, f"tt{i}_pred.png")
                fig, ax = plt.subplots(figsize=[5, 5], dpi=600)
                plot_tt_vf(self.tt_fin[i - 1], ax, plot_res=tt_plot_res)
                fig.savefig(fname, dpi=600, bbox_inches="tight", pad_inches=0)

            print(f"Transformation tensor plots saved to: {save_path}")
        except:
            print("Writing aborted.")

    def export_Obs_plot(self):
        """save the final transformation tensor plots to the selected folder."""
        fs = FileSelector(self, "Folder")

        try:
            text, ok = QtWidgets.QInputDialog().getText(
                self,
                "Save Sequence",
                "The list of the obstacles to be plotted (separated by ','):",
                QtWidgets.QLineEdit.Normal,
                "0, 1, 2, 3, 4, 5, 6, 7, 8",
            )
        except:
            print("Wrong obstacle list!")
        try:
            assert ok
            obs_no_list = np.fromstring(text, dtype=np.int8, sep=",")
            print(f"obstacle no:{obs_no_list}")
            if fs.exec_():
                save_path = fs.selectedFiles()[0]
        except:
            print("File not selected.")
        try:
            for i in tqdm(obs_no_list):
                fig, ax = create_obstacle_figure()
                if i == 0:
                    fname = os.path.join(save_path, f"Obs_temp.png")
                    plot_obstacle(self.obs_coord_tmp, self.obs_pos_tmp, ax)
                else:
                    fname = os.path.join(save_path, f"Obs{i}.png")
                    plot_obstacle(
                        self.obs_coord_fin[i - 1], self.obs_pos_fin[i - 1], ax
                    )
                fig.savefig(fname, dpi=600, bbox_inches="tight", pad_inches=0)

            print(f"Obstacle plots saved to: {save_path}")
        except:
            print("Writing aborted.")


class FileSelector(QtWidgets.QFileDialog):
    def __init__(self, parent, mode: str):
        super().__init__(parent=parent)
        self.setWindowTitle(mode)
        self.setNameFilter(str("Text files (*.txt)"))
        self.setViewMode(QtWidgets.QFileDialog.Detail)

        if mode == ("Folder"):
            self.setFileMode(QtWidgets.QFileDialog.Directory)
        elif mode == ("Load Step" or "Load Sequence"):
            self.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        else:
            self.setFileMode(QtWidgets.QFileDialog.AnyFile)
