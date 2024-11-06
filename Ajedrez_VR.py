from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
from random import choice
import time

# Iniciar tkinter
ventana = Tk()

# Tamaño de la ventana de la computadora
ancho_pantalla = ventana.winfo_screenwidth()  # 1920
altura_pantalla = ventana.winfo_screenheight()  # 1080
ventana.geometry(f"{int(ancho_pantalla)}x{int(altura_pantalla*0.9)}+0+0")
ventana.config(bg="peach puff")

# Titulo
ventana.title("Ajedrez")
# Metodo para que no pueda modificar el tamaño de la ventana
ventana.resizable(0, 0)

# Etiqueta titulo

etiqueta_titulo = Label(ventana, text='Ajedrez', fg='snow', font=("Franklin Gothic Demi", 40),
                        bg="burlywood4", width=70)
etiqueta_titulo.place(relx=0.5, rely=0.04, anchor=CENTER)

# Panel camara

panel_camara = Frame(ventana, relief=FLAT,bg="peach puff")
panel_camara.place(relx=0.24, rely=0.48, anchor=CENTER)

# Panel de configuración

panel_configuracion = Frame(ventana,relief=FLAT,width=300, height=100,bg="peach puff")
panel_configuracion.place(x=ancho_pantalla*0.648, y=altura_pantalla*0.1)

# Panel de comandos

ancho_comandos = 460
altura_comandos = 80

panel_comandos = Canvas(ventana,relief=FLAT,width=ancho_comandos, height=altura_comandos,bg="gray90")
panel_comandos.place(x=ancho_pantalla*0.675, y=altura_pantalla*0.78)

# Tablero 2d

ancho_tablero = 250
altura_tablero = 250

tablero = Canvas(ventana, width=ancho_tablero, height=altura_tablero)
tablero.place(x=ancho_pantalla*0.4845, y=altura_pantalla*0.3)

# Indicaciones

ancho_indicaciones = 200
altura_indicaciones = 80

indicaciones = Canvas(ventana,relief=FLAT,width=ancho_indicaciones,height=altura_indicaciones,bg="peach puff",highlightbackground="black", highlightthickness=2)
indicaciones.place(x=ancho_pantalla*0.5,y=altura_pantalla*0.63)

imagen = PhotoImage(file="tablero.png")
tablero_2d = imagen.subsample(int(imagen.width() / ancho_tablero), int(imagen.height() / altura_tablero))  # Redimensionar la imagen al tamaño del Canva
tablero.create_image(0, 0, anchor=NW, image=tablero_2d)

# Mostrar ganador

ganador = Canvas(ventana,relief=FLAT,width=ancho_indicaciones,height=altura_indicaciones,bg="peach puff",highlightbackground="peach puff")
ganador.place(x=ancho_pantalla*0.5,y=altura_pantalla*0.78)

# Peones eliminados

ancho_eliminados = 700
altura_eliminados = 80

peones_fuera= Canvas(ventana,relief=FLAT,width=ancho_eliminados,height=altura_eliminados,bg="peach puff",highlightbackground="black", highlightthickness=2)
peones_fuera.place(x=ancho_pantalla*0.015,y=altura_pantalla*0.78)

# Rango de colores

rojoClaro1 = np.array([0,100,20],np.uint8)
rojo0scuro1 = np.array([5,255,255],np.uint8)

rojoClaro2 = np.array([175,100,20],np.uint8)
rojo0scuro2 = np.array([179,255,255],np.uint8)

verdeClaro1 = np.array([40,100,20],np.uint8)
verdeOscuro1 = np.array([80,255,255],np.uint8)

# Seleccionar camara

camara = cv2.VideoCapture(0)

# Posiciones iniciales de los peones virtuales

peon_1 = 9
peon_2 = 11
peon_3 = 13
peon_4 = 15

indice_peones_rival = [peon_1,peon_2,peon_3,peon_4]

# Matriz para enumerar las casillas

matriz_enumerada = np.zeros((8,8))

conteo = 0

for r in range(0, 8):
    for c in range(0, 8):
        matriz_enumerada[r][c] = conteo
        conteo += 1


# Tiempo

inicio = time.time()

tiempo_transcurrido = 0

tiempo_maximo = 10

lapso = 0

# Matrices

matriz_peones = None
matriz_guardada = None
matriz_actual = None

def mascaras(frame):

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Aplicación de mascara color 1
    verdeClaro = np.array([int(Hmin.get()), int(Smin.get()), int(Vmin.get())], np.uint8)
    verdeOscuro = np.array([int(Hmax.get()), int(Smax.get()), int(Vmax.get())], np.uint8)

    maskGreen = cv2.inRange(frameHSV, verdeClaro, verdeOscuro)
    maskVerde1 = cv2.inRange(frameHSV, verdeClaro1, verdeOscuro1)

    #maskGreen = cv2.add(maskVerde, maskVerde1)

    # maskGreen = cv2.dilate(maskGreen,kernel,iterations=1)
    maskGreen = cv2.erode(maskGreen, kernel, iterations=1)

    # Aplicación de mascara color 2

    rojoClaro = np.array([int(Hmin1.get()), int(Smin1.get()), int(Vmin1.get())], np.uint8)
    rojo0scuro = np.array([int(Hmax1.get()), int(Smax1.get()), int(Vmax1.get())], np.uint8)

    maskRed = cv2.inRange(frameHSV, rojoClaro, rojo0scuro)
    maskRojo1 = cv2.inRange(frameHSV, rojoClaro1, rojo0scuro1)
    maskRojo2 = cv2.inRange(frameHSV, rojoClaro2, rojo0scuro2)

    #maskRed = cv2.add(maskRojo, maskRojo1, maskRojo2)

    #maskRed = cv2.dilate(maskRed,kernel,iterations=1)
    maskRed = cv2.erode(maskRed, kernel, iterations=1)

    # Aplicacion de mascara para detectar el peon

    azulClaro = np.array([int(Hmin2.get()), int(Smin2.get()), int(Vmin2.get())], np.uint8)
    azul0scuro = np.array([int(Hmax2.get()), int(Smax2.get()), int(Vmax2.get())], np.uint8)

    maskblue = cv2.inRange(frameHSV, azulClaro, azul0scuro)

    return maskGreen,maskRed,maskblue
def deteccion_color(frame,mask,color1):

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coordenadas = []

    for c in contornos:
        area = cv2.contourArea(c)
        if area > minArea:
            M = cv2.moments(c)
            if (M["m00"] == 0):
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            nuevoContorno = cv2.convexHull(c)
            cv2.drawContours(frame, [nuevoContorno], 0, color1, -1)
            coordenadas.append((y,x,area))

    # Ordenar las coordenadas primero por la coordenada x y luego por la coordenada y
    coordenadas.sort(key=lambda coord: (coord[0], coord[1]))

    return coordenadas
def enumerar_casillas(coordenadas_b,coodernadas_n,frame):

    contador = 0
    coodernadas_totales = coordenadas_b + coodernadas_n
    coodernadas_totales.sort(key=lambda coord: (coord[0], coord[1]))
    for y,x,area in coodernadas_totales:
        cv2.putText(frame, f"{contador}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)
        contador += 1
    return coodernadas_totales
def peones_jugador(frame,mask):

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Leer la imagen del peón con canal alfa
    peon = cv2.imread('peon_jugador.png', cv2.IMREAD_UNCHANGED)

    coordenadas_peones = []

    # Iterar sobre los contornos detectados
    for c in contornos:
        area = cv2.contourArea(c)
        if area > minArea:
            # Calcular el centroide del contorno
            M = cv2.moments(c)
            if M["m00"] == 0:
                M["m00"] = 1
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

            coordenadas_peones.append((y,x,area))

    # Ordenar las coordenadas primero por la coordenada x y luego por la coordenada y
    coordenadas_peones.sort(key=lambda coord: (coord[0], coord[1]))

    for y,x,area in coordenadas_peones:

        # Redimensionar la imagen del peón en función del área del contorno
        escala = np.sqrt(area / (peon.shape[0] * peon.shape[1]))
        resized_peon = cv2.resize(peon, (0, 0), fx=escala*4.5, fy=escala*4.5)

        # Calcular las coordenadas donde estara el peón
        x_inicio = max(0, x - resized_peon.shape[1] // 2)
        y_inicio = max(0, y - int(resized_peon.shape[0]/1.3))
        x_fin = min(frame.shape[1], x_inicio + resized_peon.shape[1])
        y_fin = min(frame.shape[0], y_inicio + resized_peon.shape[0])

        # Verificar si la imagen redimensionada del peón excede las dimensiones de la pantalla
        if x_fin - x_inicio <= 0 or y_fin - y_inicio <= 0:
            continue

        # Ajustar las dimensiones de la imagen
        peon_resized = resized_peon[:y_fin - y_inicio, :x_fin - x_inicio]

        # Extraer la información de transparencia del canal(3) alpha de la imagen y dividirlo por 255.0, para obtener un rango entre 0.0 y 1.0
        peon_alpha = peon_resized[:, :, 3] / 255.0

        # Formula de suporposicion   A => Fondo      B => Imagen(peon)      alpha = información del fondo transparente
        # Resultado = A*(1−alfa)×A + B*(alfa)

        for c in range(0, 3):
            frame[y_inicio:y_fin, x_inicio:x_fin, c] = (frame[y_inicio:y_fin, x_inicio:x_fin, c] * (1 - peon_alpha) +peon_resized[:, :, c] * peon_alpha)

    panel_comandos.delete("text1")

    if len(coordenadas_peones) < 4:
        panel_comandos.create_text(ancho_comandos*0.5, altura_comandos*0.5, text="Necesitas más piezas", font=('Dosis', 15, 'bold'), tags="text1")
    elif len(coordenadas_peones) == 4:
        panel_comandos.create_text(ancho_comandos*0.5, altura_comandos*0.5, text="Tienes todas las piezas. ¡Puedes jugar!", font=('Dosis', 15, 'bold'), tags="text1")
    elif len(coordenadas_peones) > 4:
        panel_comandos.create_text(ancho_comandos*0.5, altura_comandos*0.5, text="Hay demasidadas piezas", font=('Dosis', 15, 'bold'),tags="text1")

    return coordenadas_peones
def posicion_maquina(coordenadas_totales):

    peones_rival=[]

    matriz_peones = np.zeros((8, 8))

    if coordenadas_totales is not None and indice_peones_rival is not None:
        if max(indice_peones_rival) < len(coordenadas_totales):

            for peon in indice_peones_rival:

                if peon < len(coordenadas_totales):
                    peones_rival.append(coordenadas_totales[peon])

            for p in indice_peones_rival:
                for r in range(0,8):
                    for c in range(0,8):
                        elemento = matriz_enumerada[r][c]
                        if elemento == p:
                            matriz_peones[r][c] = 2

    return peones_rival,matriz_peones
def posicion_peon(coordenadas_peones,coordenadas_totales,matriz_peones):

    posiciones = []

    if coordenadas_totales is not None:
        for y,x,area in coordenadas_peones:
            errores = []
            for y_1,x_1,area_1 in coordenadas_totales:
                x_error = abs(x_1-x)
                y_error = abs(y_1-y)
                error = x_error + y_error
                errores.append(error)

            posiciones.append(errores.index(min(errores)))

        for p in posiciones:
            for r in range(0,8):
                for c in range(0,8):
                    elemento = matriz_enumerada[r][c]
                    if elemento == p:
                        matriz_peones[r][c] = 1

        return matriz_peones
def peones_maquina(peones_rival,frame):

    peon = cv2.imread('peon_rival.png', cv2.IMREAD_UNCHANGED)

    for y,x,area in peones_rival:

        # Redimensionar la imagen del peón en función del área del contorno
        escala = np.sqrt(area / (peon.shape[0] * peon.shape[1]))
        resized_peon = cv2.resize(peon, (0, 0), fx=escala*3.2, fy=escala*3.2)

        # Calcular las coordenadas donde estara el peón
        x_inicio = max(0, x - resized_peon.shape[1] // 2)
        y_inicio = max(0, y - int(resized_peon.shape[0] / 1.3))
        x_fin = min(frame.shape[1], x_inicio + resized_peon.shape[1])
        y_fin = min(frame.shape[0], y_inicio + resized_peon.shape[0])

        # Verificar si la imagen redimensionada del peón excede las dimensiones de la pantalla
        if x_fin - x_inicio <= 0 or y_fin - y_inicio <= 0:
            continue

        # Ajustar las dimensiones de la imagen
        peon_resized = resized_peon[:y_fin - y_inicio, :x_fin - x_inicio]

        # Extraer la información de transparencia del canal alpha de la imagen y dividirlo por 255.0, para obtener un rango entre 0.0 y 1.0
        peon_alpha = peon_resized[:, :, 3] / 255.0

        # Formula de suporposicion   A => Fondo      B => Imagen(peon)      alpha = información del fondo transparente
        # Resultado = A*(1−alfa)×A + B*(alfa)

        for c in range(0, 3):
            frame[y_inicio:y_fin, x_inicio:x_fin, c] = (frame[y_inicio:y_fin, x_inicio:x_fin, c] * (1 - peon_alpha) + peon_resized[:, :,c] * peon_alpha)
def tablero_virtual(matriz_peones):

    tablero.delete("circulo1")
    tablero.delete("circulo2")

    if matriz_peones is not None:
        for r in range(0,8):
            for c in range(0,8):
                elemento = matriz_peones[r][c]
                if elemento == 2:
                    x1 = (c + 0.1) * (ancho_tablero / 8)
                    y1 = (r + 0.1) * (altura_tablero / 8)
                    x2 = (c + 0.9) * (ancho_tablero / 8)
                    y2 = (r + 0.9) * (altura_tablero / 8)
                    tablero.create_oval(x1, y1, x2, y2, outline="white", fill="black",tags="circulo1")
                elif elemento == 1:
                    x1 = (c + 0.1) * (ancho_tablero / 8)
                    y1 = (r + 0.1) * (altura_tablero / 8)
                    x2 = (c + 0.9) * (ancho_tablero / 8)
                    y2 = (r + 0.9) * (altura_tablero / 8)
                    tablero.create_oval(x1, y1, x2, y2, outline="black", fill="white",tags="circulo2")
def ganador_juego(matriz_peones):

    for r_g in range(0, 8):
        for c_g in range(0, 8):
            campeon = matriz_peones[r_g][c_g]

            if r_g == 0:
                if campeon == 1:
                    print("Bien")
                    ganador.create_text(ancho_indicaciones * 0.5,
                                        altura_indicaciones * 0.5,
                                        text="Ganador = Humano",
                                        font=('Dosis', 15, 'bold'), tags="text3")
            elif r_g == 7:
                if campeon == 2:
                    ganador.create_text(ancho_indicaciones * 0.5,
                                        altura_indicaciones * 0.5,
                                        text="Ganador = Máquina",
                                        font=('Dosis', 15, 'bold'), tags="text3")

def comer(indice,matriz_enu,matriz_peon):

    for peon_comer in indice:
        for p_r in range(0, 8):
            for p_c in range(0, 8):
                peon_p = matriz_enu[p_r][p_c]
                if peon_p == peon_comer:
                    if p_r + 1 < 8:
                        if p_c - 1 >= 0:
                            izquierda_comer = matriz_peon[p_r + 1][p_c - 1]
                            if izquierda_comer == 1:
                                return peon_comer
                        if p_c + 1 < 8:
                            derecha_comer = matriz_peon[p_r + 1][p_c + 1]
                            if derecha_comer == 1:
                                return peon_comer

    peon_comer = choice(indice_peones_rival)

    return peon_comer

def juego(matriz_peones,matriz_guardada):

    if matriz_peones is not None:

        if matriz_guardada is not None:

            # Saber si el jugador a cambiado de lugar una pieza

            for renglon in range(0,8):
                for columna in range(0,8):

                    elemento_actual = matriz_peones[renglon][columna]
                    elemento_guardado = matriz_guardada[renglon][columna]

                    if elemento_guardado != elemento_actual:

                        # Eliminar piezas de la máquina

                        for p in indice_peones_rival:
                            for r in range(0,8):
                                for c in range(0,8):
                                    elemento = matriz_enumerada[r][c]
                                    if elemento == p:
                                        elemento2 = matriz_peones[r][c]
                                        if elemento2 == 1:
                                            indice_peones_rival.remove(p)


                        # Movimientos de la máquina

                        if indice_peones_rival:

                            peon_seleccionado = comer(indice_peones_rival,matriz_enumerada,matriz_peones)

                            for r in range(0, 8):
                                for c in range(0, 8):
                                    elemento = matriz_enumerada[r][c]
                                    if elemento == peon_seleccionado:

                                        if r + 1 < 8:  # Verificar si el movimiento hacia abajo está dentro de los límites
                                            if c - 1 >= 0:  # Verificar si el movimiento a la izquierda está dentro de los límites
                                                izquierda = matriz_peones[r + 1][c - 1]
                                                if izquierda == 1:
                                                    matriz_peones[r + 1][c - 1] = matriz_peones[r][c]
                                                    nueva_posicion = matriz_enumerada[r + 1][c - 1]
                                                    matriz_peones[r][c] = 0
                                                    indice = indice_peones_rival.index(peon_seleccionado)
                                                    indice_peones_rival[indice] = int(nueva_posicion)

                                                    ganador_juego(matriz_peones)

                                                    return matriz_peones

                                            if c + 1 < 8:  # Verificar si el movimiento a la derecha está dentro de los límites
                                                derecha = matriz_peones[r + 1][c + 1]
                                                if derecha == 1:
                                                    matriz_peones[r + 1][c + 1] = matriz_peones[r][c]
                                                    nueva_posicion = matriz_enumerada[r + 1][c + 1]
                                                    matriz_peones[r][c] = 0
                                                    indice = indice_peones_rival.index(peon_seleccionado)
                                                    indice_peones_rival[indice] = int(nueva_posicion)

                                                    ganador_juego(matriz_peones)

                                                    return matriz_peones

                                        # Si no hay movimiento válido a la izquierda o derecha, mover hacia abajo
                                        if r + 1 < 8:  # Verificar si el movimiento hacia abajo está dentro de los límites
                                            matriz_peones[r + 1][c] = matriz_peones[r][c]
                                            nueva_posicion = matriz_enumerada[r + 1][c]
                                            matriz_peones[r][c] = 0
                                            indice = indice_peones_rival.index(peon_seleccionado)
                                            indice_peones_rival[indice] = int(nueva_posicion)

                                            ganador_juego(matriz_peones)

                                            return matriz_peones

            return matriz_peones
def mostrar_video(frame,mask1,mask2,mask3):

    # Mostrar video en tkinter
    frame = cv2.resize(frame, (0, 0), fx=1.1, fy=1.1)  # Redimensiona la frame para hacerla más grande
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2.COLOR_BGR2RGB => Transformar a formato BGR
    img = Image.fromarray(img)  # La frame se convierte en un array de pixeles los cuales ya puede ser manipulados por python
    tkimagen = ImageTk.PhotoImage(img)
    video.configure(image=tkimagen)
    video.image = tkimagen

    mask_color1 = cv2.resize(mask1, (0, 0), fx=0.25, fy=0.25)
    img1 = Image.fromarray(mask_color1)
    tkimage1 = ImageTk.PhotoImage(img1)
    video_color1.configure(image=tkimage1)
    video_color1.image = tkimage1

    mask_color2 = cv2.resize(mask2, (0, 0), fx=0.25, fy=0.25)
    img2 = Image.fromarray(mask_color2)
    tkimage2 = ImageTk.PhotoImage(img2)
    video_color2.configure(image=tkimage2)
    video_color2.image = tkimage2

    mask_color3 = cv2.resize(mask3, (0, 0), fx=0.25, fy=0.25)
    img3 = Image.fromarray(mask_color3)
    tkimage3 = ImageTk.PhotoImage(img3)
    video_color3.configure(image=tkimage3)
    video_color3.image = tkimage3

    ventana.after(10, abrir_camara)  # Llamar la funcion abrir camara despues de 10 milisegundos
def peones_eliminados(peones_rival,peones_jugador):

    peones_fuera.delete("peon_r_1")
    peones_fuera.delete("peon_r_2")
    peones_fuera.delete("peon_r_3")
    peones_fuera.delete("peon_r_4")

    if len(peones_rival)==3:
        peones_fuera.create_oval(10, 10, 70, 70, outline="white", fill="black",tags="peon_r_1")
    elif len(peones_rival)==2:
        peones_fuera.create_oval(10, 10, 70, 70, outline="white", fill="black", tags="peon_r_1")
        peones_fuera.create_oval(80, 10, 140, 70, outline="white", fill="black", tags="peon_r_2")
    elif len(peones_rival)==1:
        peones_fuera.create_oval(10, 10, 70, 70, outline="white", fill="black", tags="peon_r_1")
        peones_fuera.create_oval(80, 10, 140, 70, outline="white", fill="black", tags="peon_r_2")
        peones_fuera.create_oval(150, 10, 210, 70, outline="white", fill="black", tags="peon_r_3")
    elif len(peones_rival) == 0:
        peones_fuera.create_oval(10, 10, 70, 70, outline="white", fill="black", tags="peon_r_1")
        peones_fuera.create_oval(80, 10, 140, 70, outline="white", fill="black", tags="peon_r_2")
        peones_fuera.create_oval(150, 10, 210, 70, outline="white", fill="black", tags="peon_r_3")
        peones_fuera.create_oval(220, 10, 280, 70, outline="white", fill="black", tags="peon_r_3")

    peones_fuera.delete("peon_j_1")
    peones_fuera.delete("peon_j_2")
    peones_fuera.delete("peon_j_3")
    peones_fuera.delete("peon_j_4")

    if len(peones_jugador)==0:
        peones_fuera.create_oval(410, 10, 470, 70, outline="black", fill="white",tags="peon_j_1")
        peones_fuera.create_oval(480, 10, 540, 70, outline="black", fill="white", tags="peon_j_2")
        peones_fuera.create_oval(550, 10, 610, 70, outline="black", fill="white", tags="peon_j_3")
        peones_fuera.create_oval(620, 10, 680, 70, outline="black", fill="white", tags="peon_j_4")
    elif len(peones_jugador)==1:
        peones_fuera.create_oval(480, 10, 540, 70, outline="black", fill="white", tags="peon_j_2")
        peones_fuera.create_oval(550, 10, 610, 70, outline="black", fill="white", tags="peon_j_3")
        peones_fuera.create_oval(620, 10, 680, 70, outline="black", fill="white", tags="peon_j_4")
    elif len(peones_jugador)==2:
        peones_fuera.create_oval(550, 10, 610, 70, outline="black", fill="white", tags="peon_j_3")
        peones_fuera.create_oval(620, 10, 680, 70, outline="black", fill="white", tags="peon_j_4")
    elif len(peones_jugador) == 3:
        peones_fuera.create_oval(620, 10, 680, 70, outline="black", fill="white", tags="peon_j_4")
def boton_jugar():

    if movimiento.get() == 1:
        comenzar = True
        return comenzar
    else:
        comenzar = False
        return comenzar
def abrir_camara():

    global lapso, matriz_peones, matriz_guardada, matriz_actual

    ret,frame = camara.read()

    if ret:

        # Cronometrar tiempo en un intervalo de 0 a 5 segundos

        tiempo_actual = time.time()

        tiempo_transcurrido = tiempo_actual - inicio

        intervalo = tiempo_transcurrido - lapso

        if intervalo >= tiempo_maximo:

            lapso = tiempo_transcurrido

        # Funciones del programa

        maskGreen,maskRed,maskBlue = mascaras(frame)

        coordenadas_b = deteccion_color(frame,maskGreen, (255, 255, 255))
        coordenadas_n = deteccion_color(frame,maskRed,(0,0,0))

        coordernadas_totales = enumerar_casillas(coordenadas_b,coordenadas_n,frame)

        coordenadas_peones = peones_jugador(frame,maskBlue)

        comenzar = boton_jugar()

        if comenzar == True:

            peones_rival,matriz_peones = posicion_maquina(coordernadas_totales)

            matriz_peones = posicion_peon(coordenadas_peones,coordernadas_totales,matriz_peones)

            peones_maquina(peones_rival,frame)

            if 0 < intervalo <= 0.2:

                matriz_guardada = matriz_peones

            elif 0.3 < intervalo <= 8.9:

                matriz_actual = matriz_peones
                indicaciones.delete("text2")
                indicaciones.create_text(ancho_indicaciones * 0.5, altura_indicaciones * 0.5,
                                         text="Es tu turno",
                                         font=('Dosis', 15, 'bold'), tags="text2")

            elif 9 < intervalo < 9.9:
                indicaciones.delete("text2")
                indicaciones.create_text(ancho_indicaciones * 0.5, altura_indicaciones * 0.5,
                                         text="Pensando",
                                         font=('Dosis', 15, 'bold'), tags="text2")


            elif 9.9 < intervalo <= 10:
                """
                indicaciones.delete("text2")
                indicaciones.create_text(ancho_indicaciones * 0.5, altura_indicaciones * 0.5,
                                         text="Es mi turno",
                                         font=('Dosis', 15, 'bold'), tags="text2")"""
                matriz_peones = juego(matriz_actual,matriz_guardada)
                time.sleep(0.10)

            tablero_virtual(matriz_peones)

            peones_eliminados(indice_peones_rival,coordenadas_peones)

        mostrar_video(frame,maskGreen,maskRed,maskBlue)
def RangoHSV(int):
    Hmin.set(sliderHmin.get())
    Hmax.set(sliderHmax.get())
    Smin.set(sliderSmin.get())
    Smax.set(sliderSmax.get())
    Vmin.set(sliderVmin.get())
    Vmax.set(sliderVmax.get())
def RangoHSV1(int):
    Hmin1.set(sliderHmin1.get())
    Hmax1.set(sliderHmax1.get())
    Smin1.set(sliderSmin1.get())
    Smax1.set(sliderSmax1.get())
    Vmin1.set(sliderVmin1.get())
    Vmax1.set(sliderVmax1.get())
def RangoHSV2(int):
    Hmin2.set(sliderHmin2.get())
    Hmax2.set(sliderHmax2.get())
    Smin2.set(sliderSmin2.get())
    Smax2.set(sliderSmax2.get())
    Vmin2.set(sliderVmin2.get())
    Vmax2.set(sliderVmax2.get())
def mover_pieza():
    movimiento.set(1)

#numImagen = IntVar()
#numImagen.set(0)

# Rango del 1er color
Hmin = IntVar()
Hmax = IntVar()
Smin = IntVar()
Smax = IntVar()
Vmin = IntVar()
Vmax = IntVar()

# Rango del 2do color
Hmin1 = IntVar()
Hmax1 = IntVar()
Smin1 = IntVar()
Smax1 = IntVar()
Vmin1 = IntVar()
Vmax1 = IntVar()

# Rango de deteccion del peon
Hmin2 = IntVar()
Hmax2 = IntVar()
Smin2 = IntVar()
Smax2 = IntVar()
Vmin2 = IntVar()
Vmax2 = IntVar()

kernel = np.ones((5,5),np.uint8) # Nucleo

minArea = 100  # Area minima para considerar que es un objeto

video = Label(panel_camara,bg="gray26",bd=10)
video.grid(row=0)

# Slinders para el 1er color

video_color1 = Label(panel_configuracion,bg="burlywood4")
video_color1.grid(row=0,column=0,padx=10,pady=20)

sliderHmin = Scale(panel_configuracion,label = 'Hmin Green', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmin.grid(row=1,column=0)
sliderHmin.set(0)

sliderSmin = Scale(panel_configuracion,label = 'Smin', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmin.grid(row=2,column=0)
sliderSmin.set(170)

sliderVmin = Scale(panel_configuracion,label = 'Vmin', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmin.grid(row=3,column=0)
sliderVmin.set(0)

espacio = Label(panel_configuracion, text=" ", bg="peach puff")
espacio.grid(row=4, column=0)

sliderHmax = Scale(panel_configuracion,label = 'Hmax Green', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmax.grid(row=5,column=0)
sliderHmax.set(164)

sliderSmax = Scale(panel_configuracion,label = 'Smax', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmax.grid(row=6,column=0)
sliderSmax.set(255)

sliderVmax = Scale(panel_configuracion,label = 'Vmax', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmax.grid(row=7,column=0)
sliderVmax.set(255)

# Slinders para el 2do color

video_color2 = Label(panel_configuracion,bg="burlywood4")
video_color2.grid(row=0,column=1)

sliderHmin1 = Scale(panel_configuracion,label = 'Hmin Red', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmin1.grid(row=1,column=1)
sliderHmin1.set(0)

sliderSmin1 = Scale(panel_configuracion,label = 'Smin', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmin1.grid(row=2,column=1)
sliderSmin1.set(0)

sliderVmin1 = Scale(panel_configuracion,label = 'Vmin', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmin1.grid(row=3,column=1)
sliderVmin1.set(130)

espacio = Label(panel_configuracion, text=" ", bg="peach puff")
espacio.grid(row=4, column=1)

sliderHmax1 = Scale(panel_configuracion,label = 'Hmax Red', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmax1.grid(row=5,column=1)
sliderHmax1.set(255)

sliderSmax1 = Scale(panel_configuracion,label = 'Smax', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmax1.grid(row=6,column=1)
sliderSmax1.set(153)

sliderVmax1 = Scale(panel_configuracion,label = 'Vmax', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmax1.grid(row=7,column=1)
sliderVmax1.set(255)

# Slinders para peon

video_color3 = Label(panel_configuracion,bg="burlywood4")
video_color3.grid(row=0,column=2,padx=10,pady=20)

sliderHmin2 = Scale(panel_configuracion,label = 'Hmin Peon', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV2,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmin2.grid(row=1,column=2)
sliderHmin2.set(150)

sliderSmin2 = Scale(panel_configuracion,label = 'Smin', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV2,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmin2.grid(row=2,column=2)
sliderSmin2.set(0)

sliderVmin2 = Scale(panel_configuracion,label = 'Vmin', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV2,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmin2.grid(row=3,column=2)
sliderVmin2.set(0)

espacio = Label(panel_configuracion, text=" ", bg="peach puff")
espacio.grid(row=4, column=2)

sliderHmax2 = Scale(panel_configuracion,label = 'Hmax Peon', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV2,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmax2.grid(row=5,column=2)
sliderHmax2.set(255)

sliderSmax2 = Scale(panel_configuracion,label = 'Smax', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV2,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmax2.grid(row=6,column=2)
sliderSmax2.set(255)

sliderVmax2 = Scale(panel_configuracion,label = 'Vmax', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV2,length=150,
                bg="azure4",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmax2.grid(row=7,column=2)
sliderVmax2.set(147)

# Botón mover

movimiento = IntVar()
movimiento.set(0)

boton = Button(ventana, text="Jugar", command=mover_pieza,bg="SlateGray1",font=('Dosis', 18, 'bold'))
boton.place(x=ancho_pantalla*0.49, y=altura_pantalla*0.22)


# Llamar la funcion abrir camara despues de un 10 milisegundos

ventana.after(10,abrir_camara)

ventana.mainloop()


