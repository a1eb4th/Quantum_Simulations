# Plantilla LaTeX para el TFG

(enviar comentarios y/o sugerencias a [orestes.mas@upc.edu](mailto:orestes.mas@upc.edu))

## Idiomas

Esta plantilla de TFG permite generar el PDF en 3 idiomas diferentes: catalán, castellano e inglés. Para seleccionar el idioma de generación, consulte la sección de «Personalización».

## Requisitos

Para poder compilar el documento y obtener un PDF final, necesitarás una instalación moderna y funcional de TeX. Puedes instalarlo manualmente, pero es mucho más fácil instalar una **distribución de TeX precompilada**.

1. **Para sistemas GNU/Linux**, la distribución [TeX Live][1] es la predominante, aunque hace algunos años también se portó la distribución [MiKTeX][3] a sistemas GNU/Linux. La mayoría de las distribuciones GNU/Linux, como Ubuntu, Mint, Debian o Fedora, ya tienen TeX Live empaquetado y listo para instalar, así que consulta tu gestor de paquetes.
2. **Para ordenadores con macOS**, puedes instalar [MacTeX][2], que es esencialmente TeX Live para Mac.
3. **Para Windows**, puedes elegir entre [MiKTeX][3] o [TeX Live]. La primera de ellas es más popular entre los usuarios de Windows, probablemente porque ha estado disponible durante más tiempo.
4. **En línea**: Algunas personas comienzan a dar sus primeros pasos en LaTeX utilizando un servicio en línea como [Overleaf][4]. Aunque estos servicios pueden ser útiles para reducir la barrera de entrada, ten en cuenta que son servicios "freemium", es decir, la versión gratuita tiene algunas limitaciones (normalmente en longitud del documento y tiempo de procesamiento) y es necesario actualizar a una versión de pago para eliminar estas limitaciones.

[1]: https://tug.org/texlive/
[2]: https://tug.org/mactex/
[3]: https://miktex.org/
[4]: https://www.overleaf.com/

## Aspectos a tener en cuenta

LaTeX es un programa muy antiguo que ha ido evolucionando con el tiempo. Además, se han conservado versiones antiguas para preservar la compatibilidad. Esto significa que actualmente existen 2 formas principales de generar tu documento:

1. **pdflatex**: Es la forma "antigua" que compila más rápido, pero no admite directamente el uso de fuentes OpenType y características tipográficas avanzadas.
2. **lualatex**: Por otro lado, es un poco más lento en la compilación pero admite directamente fuentes TrueType y OpenType.

La plantilla LaTeX está diseñada y configurada para compilar correctamente de ambas maneras. Deberás elegir cuál de las dos opciones prefieres utilizar (puedes probar ambas para ver la diferencia de velocidad). Personalmente, te recomendamos la segunda opción, ya que es más moderna y probablemente prevalecerá a medio plazo.

## Generación del documento

El proceso de generación de un documento LaTeX se puede dividir en varios pasos. Aquí tienes una breve explicación del procedimiento:

1. **Edición del documento**: El usuario escribe el contenido del documento en un editor de texto. El documento está escrito en formato LaTeX, que incluye comandos y marcadores para indicar su estructura y formato. Para ayudar en esta tarea, existen editores de texto especialmente diseñados para esto, como **TeXStudio**.

2. **Compilación**: Para obtener un documento final (generalmente en formato PDF), es necesario compilar el documento LaTeX. Este proceso implica el uso de un motor LaTeX, como "pdflatex" o "lualatex", para procesar el documento y generar el resultado deseado. Durante la compilación se generan varios **archivos auxiliares** que ayudan a LaTeX a gestionar la información, como los índices, las bibliografías y las referencias cruzadas.

3. **Gestión de bibliografía**: Si se utiliza una bibliografía, LaTeX utiliza un software como "biber" para procesar las citas y generar la lista de referencias bibliográficas.

4. **Compilación múltiple**: En muchos casos, es necesario compilar el documento más de una vez para resolver las referencias cruzadas y otros elementos que requieren información adicional.

5. **Generación del documento final**: Finalmente, el motor LaTeX crea el documento final en el formato deseado, como un archivo PDF. El documento incluye todos los elementos, como texto, imágenes, tablas y gráficos, con el formato especificado en el documento LaTeX original.

El resultado es un documento de alta calidad tipográfica que sigue las normas estándar de formato y estilo de LaTeX. Este proceso de compilación permite a los autores centrarse en el contenido del documento mientras LaTeX gestiona la presentación y la tipografía de manera automática.

Por lo tanto, la primera vez que compilemos el documento, deberemos hacer lo siguiente:

```sh
pdflatex TFG.tex (repetir el comando dos o más veces para resolver las referencias cruzadas)
biber TFG        (esto genera la bibliografía)
pdflatex TFG.tex (repetir 2 o más veces para incorporar la bibliografía y resolver las referencias)
```

o:

```sh
lualatex TFG.tex (repetir el comando dos o más veces para resolver las referencias cruzadas)
biber TFG        (esto genera la bibliografía)
lualatex TFG.tex (repetir 2 o más veces para incorporar la bibliografía y resolver las referencias)
```

Una vez generado sin errores ni advertencias, si no modificamos la bibliografía, podremos actualizar únicamente el archivo .TeX principal con `lualatex/pdflatex` y listo.

## Personalización de la plantilla (IMPORTANTE !)

### Información del TFG
Para adaptar la plantilla a tus necesidades, debes **editar el archivo principal (`TFG.tex`)** e introducir tu información personal:

1. Indica el idioma del documento en la variable `\doclanguage` (línea 31).
2. Descomenta la macro `\setTitle` y coloca el título de tu TFG (línea 37).
3. Haz lo mismo con las macros `\setAuthor`, `\setAdvisor`, `\setReviewer` y `\setDate`.
4. También modifica la macro `\setDegree`, pero aquí no debes poner un texto aleatorio, ya que las variables \GRETST y \GREELEC ya se encargan de todo.
5. Cambia el texto de las diferentes secciones de agradecimientos, dedicatoria, etc.
6. Finalmente, edita el contenido de los diferentes archivos en los que se divide el documento para poner tu propio contenido, escrito en el idioma que hayas elegido.

### Cambios de estilo

Si deseas cambiar o agregar algo al estilo del documento, la mayoría de las opciones las encontrarás en el archivo `config/styles.tex`. Sin embargo, dado que se trata de un documento oficial y formal, no es conveniente cambiar la apariencia general.

### Agregar funcionalidad

Para ampliar las capacidades de LaTeX, puedes agregar paquetes al preámbulo. El lugar ideal para hacerlo es en el archivo `config/packages.tex`.