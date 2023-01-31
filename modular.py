"""
Librería principal de cálculo aritmético que realiza el trabajo de resolución de los problemas
aritméticos.
"""

from typing import *
import numpy as np
import math

NE = NOP = float('nan')


def es_primo(n: int) -> bool:
    """
    Esta función dado un entero positivo, devuelve Falso si el entero no es primo, y verdadero si sí lo es. Para ello
    se hace uso del test de Miller.
    :param n: int. Número en cuestión para ver si es primo.
    :return: bool. Si True = Es primo, si False = No es primo.
    """
    if n == 2 or n == 3:
        return True

    if n < 2 or not n & 1 or not n % 3:
        return False

    if n < 9:
        return True

    for a in range(2, min(n - 2, int(2 * math.log(n) ** 2)) + 1):
        def miller_rabin():
            exp = n - 1
            while not exp & 1:
                exp >>= 1

            x = potencia_mod_p(a, exp, n)
            if x == 1 or x == n - 1:
                return True

            while exp < n - 1:
                if potencia_mod_p(a, exp, n) == n - 1:
                    return True
                exp <<= 1
            return False
        if not miller_rabin():
            return False
    return True


def lista_primos(a: int, b: int) -> List[int] or NE:
    """
    Esta función recibe 2 enteros a y b, y devuelve la lista de primos en el intervalo [a, b). Esta función crea una
    media criba de Eratóstenes (ignorando los múltiplos de dos que son trivialmente compuestos y fáciles de omitir).
    Genera un array de booleanos, en el que de base todos los valores del array son primos (True), y a medida que se
    pruebe que cada número es primo, todos sus múltiplos se tacharan de serlo (False). Si al iterar sobre la criba el
    número no ha sido "tachado", significa que no es factorizable en primos. El máximo número de entradas
    equivale a 2.7 mil millones.

    :param a: int. Cota inferior
    :param b: int. Cota superior
    :return: List[int]. Lista de primos en [a, b).
    """

    try:
        assert b >= 2  # Esta suposición se debe a que 2 es el menor primo existente
        criba = np.ones(b // 2, dtype=bool)  # Creamos media criba (sin pares) de booleanos (True)

        for numero in range(3, int(math.sqrt(b) + 1), 2):  # Usamos la cota de primos (ambos factores no mayores a √b)
            if criba[numero // 2]:  # Si numero no está tachado (es primo). Se divide por dos al no coger los pares.
                criba[numero * numero // 2::numero] = False  # Actualiza todos los múltiplos del número en cuestión.

        lista_de_primos = [2, *2 * np.nonzero(criba)[0][1::] + 1]  # Escoge los primos
        lista_acotada = lista_de_primos[np.searchsorted(lista_de_primos, a, side="left")::]  # Acota la lista en [a,b)
        return lista_acotada
    except AssertionError:
        return NE


def factorizar(n: int, factores_dict: Dict[int, int] = None) -> Dict[int, int] or NE:
    """
    Esta función dado un número, lo factoriza en números primos. Para ello se hace uso del algoritmo de Pollard rho
    con una serie de correcciones para que la función sea determinista. La factorización que devuelve la función es
    única por el teorema fundamental de la aritmética.
    :param factores_dict: Dict[int, int]
    :param n: int. Número que se busca factorizar.
    :return: Dict[int, int]. Diccionario de claves: factores primos del número y valores: exponentes de cada factor.
             NE. Si no se puede factorizar en primos positivos.
    """
    try:
        assert n > 1

        if factores_dict is None:
            factores_dict = {}              # output diccionario

        while not n & 1:
            n >>= 1
            if 2 in factores_dict.keys():
                factores_dict[2] += 1
            else:
                factores_dict[2] = 1

        if n == 1:
            return factores_dict

        if es_primo(n):
            if n in factores_dict.keys():
                factores_dict[n] += 1
            else:
                factores_dict[n] = 1
            return factores_dict

        def g(number: int) -> int:
            return number ** 2 + 1

        x, y, d = 2, 2, 1

        while d == 1:
            x = g(x) % n
            y = g(g(y)) % n
            d = mcd((x - y) % n, n)

        if not es_primo(d):
            n //= d
            divisor = 3
            while divisor * divisor <= d:  # cota factores primos
                if d % divisor:  # si el divisor deja resto, no es factor
                    divisor += 1
                else:
                    if divisor in factores_dict.keys():
                        factores_dict[divisor] += 1
                    else:
                        factores_dict[divisor] = 1
                    d //= divisor
            factores_dict[divisor] += 1
            return factorizar(n, factores_dict)

        if d != n:
            if d in factores_dict.keys():
                factores_dict[d] += 1
            else:
                factores_dict[d] = 1
            return factorizar(n // d, factores_dict)
    except AssertionError:
        return NE


def mcd(a: int, b: int) -> int:
    """
    Esta función devuelve el máximo común divisor de dos entradas enteras. (Algoritmo Euclides)
    :param a: int. Primer número
    :param b: int. Segundo número
    :return: int. Máximo común divisor de las dos entradas.
    """
    while b:
        a, b = b, a % b
    return abs(a)


def mcd_n(nlist: List[int]) -> int:
    """
    Esta función devuelve un entero, máximo común divisor de la lista de entrada.
    :param nlist: Lista de enteros.
    :return: int. Máximo común divisor lista.
    """
    if len(nlist) == 1:
        return nlist[0]
    nlist = nlist[:-2] + [mcd(nlist[-1], nlist[-2])]
    return mcd_n(nlist)


def mcm(a: int, b: int) -> int:
    """
        Esta función devuelve el mínimo común múltiplo dados dos números enteros.
        :param a: int. Primer número
        :param b: int. Segundo número
        :return: int. Mínimo común múltiplo de las dos entradas.
    """
    return abs(a * b) // mcd(a, b)


def bezout(a: int, b: int) -> Tuple[int, int, int]:
    """
    Esta función encuentra una solución particular a la ecuación diofántica de la forma ax + by = mcd(a, b). Para ello
    emplea el algoritmo de euclides extendido.
    :param a: Primer coeficiente de la ecuación diofántica.
    :param b: Segundo coeficiente de la ecuación diofántica.
    :return: Tuple[int, int, int]. Esta tupla contiene el mcd(a, b) y los dos coeficientes de bezout respectivamente.
    """
    s1, s2 = 1, 0
    t1, t2 = 0, 1
    while b:
        cociente, residuo = divmod(a, b)
        a, b = b, residuo
        s1, s2 = s2, s1 - cociente * s2
        t1, t2 = t2, t1 - cociente * t2
    return a, s1, t1


def coprimos(a: int, b: int) -> bool:
    """
    Dados dos entradas enteras, esta función devuelve un booleano; True si ambos enteros son coprimos, o sea, su máximo
    común divisor es 1, o False en caso contrario.
    :param a: int. Primer entero
    :param b: int. Segundo entero
    :return: bool. El resultado será True si son coprimos y False si no lo son
    """
    if mcd(a, b) == 1:
        return True
    else:
        return False


def coprimos_dos_a_dos(array: List) -> bool:
    """
    Dada una lista, esta función devuelve True si todos los valores son coprimos dos a dos y False si no lo son. Para
    ello comprobamos que el producto .
    :param array: List. Lista a comprobar
    :return: bool. True si coprimos dos a dos, False en caso contrario
    """
    producto = 1
    lcm = 1
    for i in range(len(array)):
        producto *= array[i]
        lcm = mcm(array[i], lcm)

    if producto == lcm:
        return True
    else:
        return False


def potencia_mod_p(base: int, exp: int, p: int) -> int:
    """
    Esta función devuelve el menor entero positivo congruente a una potencia dada de base == base y exponente == exp.
    Esta función utiliza el algoritmo de potenciación de cuadrados. Para ello expresamos el exponente en binario base 2.
    Por ejemplo: si exp == 11 == (11)10 = (1011)2 = 2^3 + 2^1 + 2^0. Por tanto, la base elevada a su exponente sería un
    número elevado a una suma de potencias de 2. Cada vez que se eleva al cuadrado una base de exponentes binarios, el
    exponente binario se desplaza a la izquierda. Por ejemplo: x^(1)2 * x^(1)2 = x^(10)2, o x^(10)2 * x^(10)2 =
    x^(100)2; haciendo que trabajar con exponentes binarios sea muy sencillo. Si se multiplica una potencia por su base,
    simplemente se le suma uno a su exponente: x^(100) * x = x^(101)2. Nuestro objetivo es llegar al exponente deseado,
    empleando estas propiedades, empezando desde la base elevado a 1.
    :param base: int. Base entera de la potencia
    :param exp: int. Exponente entero de la potencia
    :param p: int. Módulo
    :return: int. Entero congruente módulo p a la potencia
    """
    if exp < 0:
        base = inversa_mod_p(base, p)
        exp = -exp

    res = 1
    base = base % p
    while exp > 0:
        if (exp & 1) != 0:          # si es impar
            res = (res * base) % p      # para acelerar el proceso se irán calculando los módulos progresivamente
        exp >>= 1
        base = (base ** 2) % p          # para acelerar el proceso se irán calculando los módulos progresivamente
    return int(res % p)


def inversa_mod_p(n: int, p: int) -> int or NE:
    """
    Esta función devuelve la inversa de un número dado en módulo p dado. Si las entradas enteras no son relativamente
    primas, entonces no existe una inversa, y se devolverá NE.
    :param n: int. Número del que se quiere saber su inversa módulo p
    :param p: int. Módulo
    :return: int. Entero congruente módulo p a la potencia. NE. Las entradas no son coprimas.
    """
    try:
        assert mcd(n, p) == 1
        return int(bezout(n, p)[1] % p)
    except AssertionError:
        return NE


def euler(n: int) -> int or NE:
    """
    Esta función, dado un entero, devuelve el número de primos relativos positivos menores que él. Para ello se hace uso
    de las siguientes propiedades:
        - Si p primo: phi(p) = p-1
        - Si p primo: phi(p^k) = (p-1)*p^(k-1) = p^k-p^(k-1)
        - Si a y b coprimos: phi(a*b) = phi(a) * phi(b)
    Entonces si n = p1^a1 * p2^a2 * ... * pk^ak (factores primos)(1), por corolario de estas tres propiedades:
        phi(n) = phi(p1^a1) * phi(p2^a2) * ... * phi(pk^ak) =
               = p1^a1 * (1 - 1/p1) * p2^a2 * (1 - 1/p2) * ... * pk^ak * (1 - 1/pk) =
               = n * (1 - 1/p1) * (1 - 1/p2) * ... * (1 - 1/pk)     (2)
    :param n: int. Entero positivo del que queremos hallar phi(n)
    :return: int. Número de primos relativos positivos menores que n. NE si la entrada es 0 o menor.
    """
    try:
        assert n > 0
        if n == 1:
            return 1
        totiente = n                            # output
        for factor in factorizar(n):            # (1)
            totiente -= totiente // factor      # (2)
        return totiente
    except AssertionError:
        return NE


def legendre(n: int, p: int) -> int or NOP:
    """
    El símbolo de Legendre (n, p) con p siendo primo, devuelve: 0 si n es congruente con 0 módulo p, 1 si n es un
    residuo cuadrático módulo p y -1 en caso contrario.
    :param n: int. Entero
    :param p: int. Entero primo
    :return: int. Símbolo de Legendre. NE si p no es primo.
    """
    try:
        assert es_primo(p)
        simbolo_legendre = potencia_mod_p(n, (p - 1)//2, p)
        if simbolo_legendre == p - 1:
            return -1
        return simbolo_legendre
    except AssertionError:
        return NE


def resolver_sistema_congruencias(alist: List[int], blist: List[int], plist: List[int]) -> Tuple[int, int] or NOP or NE:
    """
    Dado una lista de coeficientes a, una lista de congruencias b y una lista de módulos p; devuelve una tupla solución
    al sistema de congruencias. El segundo elemento de la tupla es el módulo de la solución, y el primero la solución
    con dicho módulo. Para ello se hace uso del teorema chino del resto.
    :param alist: List[int]. Lista de coeficientes de la x.
    :param blist: List[int]. Lista de congruencias
    :param plist: List[int]. Lista de módulos
    :return: Tuple[int, int]. Solución al sistema. NOP. Elementos lista a y b no coprimos. NE. No existe solución.
    """

    try:
        numero_ecuaciones = len(alist)
        assert coprimos_dos_a_dos(plist)

        # Hallar coeficientes c, tal que el sistema queda de la forma x = ci (mod p)
        clist = [blist[i] * inversa_mod_p(alist[i], plist[i]) if coprimos(alist[i], plist[i])
                 else blist[i]/mcd(alist[i], plist[i]) * inversa_mod_p(int(alist[i]/mcd(alist[i], plist[i])),
                 int(plist[i]/mcd(alist[i], plist[i])))
                 for i in range(numero_ecuaciones)]

        if all([isinstance(item, int) for item in clist]):
            m = math.prod(plist)

            res = 0
            for i in range(numero_ecuaciones):
                modulo_i = int(m / plist[i])
                res += clist[i] * modulo_i * inversa_mod_p(modulo_i, plist[i])
            return res % m, m
        else:
            return NE

    except AssertionError:
        return NOP


def raiz_mod_p(n: int, p: int) -> int or NE:
    """
    Esta función resuelve una ecuación de la forma x^2 = n (mod p). Para ello se utiliza el algoritmo Cipolla.
    :param n: int. Entero resultado de la ecuación.
    :param p: int. Módulo entero de la ecuación
    :return: int. Incógnita de la ecuación y raíz módulo p de n
    """
    try:
        assert es_primo(p) and legendre(n, p) != -1
        if not n % p:
            return 0

        a = 2
        while legendre(a**2 - n, p) == -1:
            a += 1

        a -= 1
        d = a**2 - n

        res = (1, 1)
        base = (a, 1)
        exp = (p + 1) // 2

        while exp > 0:
            if (exp & 1) != 0:  # si es impar
                res = multiplicacion_cuerpo_cuadratico_mod_p(res, base, d, p)
            exp >>= 1
            base = multiplicacion_cuerpo_cuadratico_mod_p(base, base, d, p)
        return res[0]

    except AssertionError:
        return NE


def multiplicacion_cuerpo_cuadratico_mod_p(x: Tuple[int, int], y: Tuple[int, int], d: int, p: int) -> Tuple[int, int]:
    resultado = (x[0] * y[0] + x[1] * y[1] * d, x[0] * y[1] + x[1] * y[0])
    return resultado[0] % p, resultado[1] % p


def ecuacion_cuadratica(a: int, b: int, c: int, p: int) -> Tuple[int, int] or NE:
    """
    Esta función recibe de entrada los enteros a, b, c y p, con p número primo, y devuelve las soluciones módulo p de la
    ecuación ax^2 + bx + c ≡ 0 (mod p).
    :param a: int. Coeficiente de x^2
    :param b: int. Coeficiente de x.
    :param c: int. Término independiente
    :param p: int. Módulo entero de la ecuación.
    :return: Tuple[int, int]. Soluciones de la ecuación
    """
    try:
        assert es_primo(p) and coprimos(a, p)
        # z^2 = (2ax + b)^2
        z1 = raiz_mod_p((b ** 2 - 4 * a * c) % p, p)
        if isinstance(z1, int):
            z2 = p - z1
            x1 = resolver_sistema_congruencias([2 * a], [z1 - b], [p])[0]
            x2 = resolver_sistema_congruencias([2 * a], [z2 - b], [p])[0]
            return x1, x2
        else:
            raise AssertionError

    except AssertionError:
        return NE
