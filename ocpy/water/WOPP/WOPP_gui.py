#!/usr/bin/env python
"""
THIS PROGRAM IS EXPERIMENTAL AND IS PROVIDED "AS IS" WITHOUT
REPRESENTATION OF WARRANTY OF ANY KIND, EITHER EXPRESS OR
IMPLIED. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF
THE PROGRAM IS WITH THE USER.
COPYRIGHT (C) 2010 BY HZG/KOF

Wolfgang Schoenfeld, 04/2010
wolfgang.schoenfeld@hzg.de

11/2010: errormargins added, bugfixes; schoenfeld
04/2013: select water abs file
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Tkinter as Tk
import tkFileDialog,  tkMessageBox
import WOPP as model

#def select_abs_file():
fn_abs1='purewater_abs_coefficients_v1.dat'
fn_abs2='purewater_abs_coefficients_v2.dat'
fn_abs3='purewater_abs_coefficients_v3.dat'
def info1():
    tkMessageBox.showinfo("Version 1", 
        """Absorption of pure water in the range 300 - 420 nm is based on data of Ed Fry and coworkers, see ATBD""")
def info2():
    tkMessageBox.showinfo("Version 2", 
    """Absorption of pure water in the range 300 - 420 nm is based on Morel et al. 2007, see ATBD""")
    
def info3():
    tkMessageBox.showinfo("Version 3", 
    """Absorption of pure water in the range 300 - 510 nm is based on Mason et al. 2016, see ATBD""")


def doplot():
    global X, Y,  Yerr,  Unit
    Tc=temp.get()
    S=sal.get()
    theta=np.array([_theta.get()])
    param=radio.get()
    dp=depol.get()
    aw=_aw.get()
    if aw%2: aw-=1
    ew=_ew.get()
    if ew%2: ew+=1
    if ew<=aw:
        tkMessageBox.showwarning( "Error","Start Wavelength must be < End Wavelength")
        return
    if aw>700:
        tkMessageBox.showwarning( "Error","Start Wavelength must be < 700 nm")
        return    
    fn_ref='computed_refri_T27_S0_180_4000nm.dat'
    wl,  std_RI,  std_RI_err = model.read_refri_std(fn_ref, aw, ew)
    if param=='refractive_index':
        refr_index,  dnswds=model.refractive_index(wl, S, Tc,  std_RI)
        X=wl
        Y=refr_index
        Yerr= std_RI_err    #TODO
        Unit=''
    elif param=='absorption1':
#        fn_abs='purewater_abs_coefficients_v1.dat'
        wl, abso,  a_err=model.absorption(aw, ew,  S,  Tc,  fn_abs1) 
        X=wl
        Y=abso
        Yerr= a_err
        Unit='[1/m]'
    elif param=='absorption2':
#        fn_abs='purewater_abs_coefficients_v2.dat'
        wl, abso,  a_err=model.absorption(aw, ew,  S,  Tc,  fn_abs2) 
        X=wl
        Y=abso
        Yerr= a_err
        Unit='[1/m]'    
    elif param=='absorption3':
#        fn_abs='purewater_abs_coefficients_v3.dat'
        wl, abso,  a_err=model.absorption(aw, ew,  S,  Tc,  fn_abs3) 
        X=wl
        Y=abso
        Yerr= a_err
        Unit='[1/m]' 
    elif param=='scattering':
        betasw,beta90sw,bsw , err_betasw, err_beta90sw,  err_bsw= model.scattering(wl,S, Tc, theta, std_RI)
        X=wl
        Y=betasw
        Yerr= err_betasw
        Unit='[1/(m*sr)]'
    elif param=='backscattering':
        betasw,beta90sw,bsw , err_betasw, err_beta90sw,  err_bsw= model.scattering(wl,S, Tc, theta, std_RI)
        X=wl
        Y=bsw / 2.
        Yerr= err_bsw / 2.
        Unit='[1/m]'
    elif param=='totalscattering':
        betasw,beta90sw,bsw , err_betasw, err_beta90sw,  err_bsw= model.scattering(wl,S, Tc, theta, std_RI)
        X=wl
        Y=bsw
        Yerr= err_bsw
        Unit='[1/m]'
    logscale=_logscale.get()   
    errormargin=_errormargin.get()
    if logscale:
        plotarea.semilogy(X, Y, 'b')
        if errormargin:
            plotarea.semilogy(X, Y+Yerr,':r',  X, Y-Yerr, ':r')
    else:
#        plotarea.cla()
        plotarea.plot(X, Y, 'b')
        if errormargin:
            plotarea.plot(X, Y+Yerr,':r',  X, Y-Yerr, ':r')
    plotarea.title.set_text(param)
    plotarea.set_xlabel('wavelength [nm]')
    canvas.show()

def clear():
    plotarea.cla()
    canvas.show()
    
def save():
    Tk.Tk().withdraw()
    fnw=tkFileDialog.asksaveasfilename()
    #print fn,  type(fn)
    global X, Y,  Yerr,  Unit
    descr='wl [nm]\t%s %s\t+/- error %s\nat T=%4.1f degC, S=%4.1f PSU, depol.ratio=%5.3f, Theta=%4.1f deg'%(radio.get(),Unit, Unit,   temp.get(), sal.get(), depol.get(),  _theta.get())
    if fnw:
        fpw=open(fnw, 'w')
        fpw.write('%s\n'%descr)
        for i in range(len(X)):
            fpw.write('%5.1f\t%15.10f\t%15.10f\n'%(X[i], Y[i], Yerr[i]))
        fpw.close()
        
def destroy(): 
    root.destroy()
#    sys.exit()    

##################################################    
root = Tk.Tk()
root.wm_title("WOPP - water optical properties processor")
root.option_add("*Dialog.msg.wrapLength", "3.7i") 

c, r=0, 0
frame_l=Tk.Frame(root).grid(column=c, row=r)

c, r=0, 0
logo = Tk.PhotoImage(file="water_radiance5.gif")
Tk.Label(frame_l, image=logo, width=200).grid(column=c, row=r, columnspan=2, sticky=Tk.N)
#c, r=0, 1
#logo1 = Tk.PhotoImage(file="logo_esa.gif")
#Tk.Label(frame_l, image=logo1, width=167).grid(column=c, row=r, columnspan=2)

c, r=0, 2
radio_frame=Tk.Frame(root)
radio_frame.grid(column=c, row=r, columnspan=2, pady=10)
radio=Tk.StringVar()
rb_refr = Tk.Radiobutton(radio_frame, variable=radio, value='refractive_index', text='refractive index')
rb_refr.grid(column=c, row=r, sticky=Tk.W)
rb_refr.select()
rb_abso = Tk.Radiobutton(radio_frame, variable=radio, value='absorption1', text='absorption coeff. Version 1')
rb_abso.grid(column=c, row=r+1, sticky=Tk.W)
info_button = Tk.Button(master=radio_frame, text='Info', command=info1).grid(column=c+1, row=r+1, sticky=Tk.E)

rb_abso1 = Tk.Radiobutton(radio_frame, variable=radio, value='absorption2', text='absorption coeff. Version 2')
rb_abso1.grid(column=c, row=r+2, sticky=Tk.W)
info_button = Tk.Button(master=radio_frame, text='Info', command=info2).grid(column=c+1, row=r+2, sticky=Tk.E)

rb_abso2 = Tk.Radiobutton(radio_frame, variable=radio, value='absorption3', text='absorption coeff. Version 3')
rb_abso2.grid(column=c, row=r+3, sticky=Tk.W)
info_button = Tk.Button(master=radio_frame, text='Info', command=info3).grid(column=c+1, row=r+3, sticky=Tk.E)

rb_scat = Tk.Radiobutton(radio_frame, variable=radio, value='scattering', text='volume scattering (theta)')
rb_scat.grid(column=c, row=r+4, sticky=Tk.W)
rb_scat = Tk.Radiobutton(radio_frame, variable=radio, value='backscattering', text='backscattering coeff.')
rb_scat.grid(column=c, row=r+5, sticky=Tk.W)
rb_scat = Tk.Radiobutton(radio_frame, variable=radio, value='totalscattering', text='total scattering coeff.')
rb_scat.grid(column=c, row=r+6, sticky=Tk.W)

c, r=0, 3
Tk.Label(frame_l,text='Start Wavelength (>=300 nm)').grid(column=c, row=r, sticky=Tk.W)
_aw=Tk.DoubleVar()
_aw.set(300)
_aw_input=Tk.Entry(frame_l, textvariable=_aw, bg='white', width=5)
_aw_input.grid(column=c+1, row=r, sticky=Tk.W)

c, r=0, 4
Tk.Label(frame_l,text='End Wavelength (<=4000 nm)').grid(column=c, row=r, sticky=Tk.W)
_ew=Tk.DoubleVar()
_ew.set(4000)
_ew_input=Tk.Entry(frame_l, textvariable=_ew, bg='white', width=5)
_ew_input.grid(column=c+1, row=r, sticky=Tk.W)

c, r=0, 5
Tk.Label(frame_l,text='Temperature (0-95 degC)').grid(column=c, row=r, sticky=Tk.W)
temp=Tk.DoubleVar()
temp.set(20)
temp_input=Tk.Entry(frame_l, textvariable=temp, bg='white', width=5)
temp_input.grid(column=c+1, row=r, sticky=Tk.W)

c, r=0, 6
Tk.Label(frame_l,text='Salinity (0-100 PSU)').grid(column=c, row=r, sticky=Tk.W)
sal=Tk.DoubleVar()
sal.set(0)
sal_input=Tk.Entry(frame_l, textvariable=sal, bg='white', width=5)
sal_input.grid(column=c+1, row=r, sticky=Tk.W)

c, r=0, 7
Tk.Label(frame_l,text='Depolarisation ratio (0.039)').grid(column=c, row=r, sticky=Tk.W)
depol=Tk.DoubleVar()
depol.set(0.039)
depol_input=Tk.Entry(frame_l, textvariable=depol, bg='white', width=5)
depol_input.grid(column=c+1, row=r, sticky=Tk.W)

c, r=0, 8
Tk.Label(frame_l,text='Theta (0-180 deg)').grid(column=c, row=r, sticky=Tk.W)
_theta=Tk.DoubleVar()
_theta.set(90)
_theta_input=Tk.Entry(frame_l, textvariable=_theta, bg='white', width=5)
_theta_input.grid(column=c+1, row=r, sticky=Tk.W)

c, r = 0, 9
_logscale=Tk.BooleanVar()
#_logscale.set(True)
cb=Tk.Checkbutton(frame_l, variable=_logscale,   text='Y-logscale').grid(column=c, row=r, sticky=Tk.W)
c, r = 0, 10
_errormargin=Tk.BooleanVar()
#_logscale.set(True)
cb=Tk.Checkbutton(frame_l, variable=_errormargin,   text='show error margins').grid(column=c, row=r, sticky=Tk.W)

############buttons##############################################
c, r=0, 11
button_frame=Tk.Frame(root)
button_frame.grid(column=c, row=r, columnspan=2)
plot_button = Tk.Button(master=button_frame, text='Plot', command=doplot).grid(column=0, row=0)
clear_button = Tk.Button(master=button_frame, text='Clear', command=clear).grid(column=1, row=0)
save_button = Tk.Button(master=button_frame, text='Save', command=save).grid(column=2, row=0)
quit_button = Tk.Button(master=button_frame, text='Quit', command=destroy).grid(column=3, row=0)

c, r=0, 12
Tk.Label(frame_l, text='(c) HZG/ESA', font=('Arial', 10,'bold' ) ).grid(column=c, row=r, columnspan=2)

##########canvas###################################
f = matplotlib.figure.Figure(figsize=(8,6))
plotarea = f.add_subplot(111)
canvas = FigureCanvasTkAgg(f, master=root)
c, r=2, 0
canvas.get_tk_widget().grid(column=c, row=r, rowspan=13)  
#fn_abs=select_abs_file()
Tk.mainloop()
