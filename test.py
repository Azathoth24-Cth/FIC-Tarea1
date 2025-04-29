Comando = input("Ingerse un comando:")

if Comando[0:5] == "START":
    if int(Comando[6:-1]) > 0 and int(Comando[6:-1]) <= 100:
        print("Valor correcto.")
    else:
        print("Valor incorrecto.")
elif Comando[0:3] == "PWM":
    if int(Comando[4:-1]) > 0 and int(Comando[4:-1]) <= 100:
        print("Valor correcto.")
    else:
        print("Valor Incorrecto.")
else:
    print("Comando incorrecto.")