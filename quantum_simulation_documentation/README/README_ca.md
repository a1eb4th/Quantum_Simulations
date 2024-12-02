# Patró LaTeX per al TFG

(envieu comentaris i/o suggerències a [orestes.mas@upc.edu](mailto:orestes.mas@upc.edu))

## Idiomes

Aquest patró de TFG permet generar el PDF en 3 idiomes diferents: català, castellà i angles. Per seleccionar l'idioma de generació vegeu l'apartat de «Personalització».

## Requisits

Per poder compilar el document i obtenir un PDF final, necessitareu una instal·lació moderna i funcional de TeX. Es pot instal·lar manualment, però és molt més fàcil instal·lar una **distribució TeX precompilada**.

1. **Per a sistemes GNU/Linux**, la distribució [TeX Live][1] és la predominant, tot i que fa alguns anys la distribució [MiKTeX][3] també es va portar a sistemes GNU/Linux. La majoria de les distribucions GNU/Linux, com ara Ubuntu, Mint, Debian o Fedora, ja tenen TeX Live empaquetat i llest per instal·lar, així que consulteu el vostre gestor de paquets.
2. **Per a ordinadors amb macOS**, pots instal·lar [MacTeX][2], que és essencialment TeX Live per a Mac.
3. **Per a Windows**, pots triar entre [MiKTeX][3] o [TeX Live]. La primera d'elles és més popular entre els usuaris de Windows, probablement perquè fa més anys que és disponible.
4. **En línia**: Algunes persones comencen a fer els seus primers passos en LaTeX utilitzant un servei en línia com [Overleaf][4]. Tot i que aquests serveis poden ser útils per baixar la barrera d'entrada, tingues en compte que són serveis _freemium_, és a dir, la versió gratuïta té algunes limitacions (normalment en longitud del document i temps de processament) i cal de fer una actualització a una versió de pagament per eliminar aquestes limitacions.

[1]: https://tug.org/texlive/
[2]: https://tug.org/mactex/
[3]: https://miktex.org/
[4]: https://www.overleaf.com/

## Per tenir en compte

El (La)TeX és un programa molt antic que ha anat evolucionant. Alhora, les versions antigues s'han volgut conservar per tal de preservar la compatibilitat. Això fa que actualment hi ha 2 maneres principals de generar el vostre document:

1. **pdflatex**: És la manera «antiga», que compila més ràpidament però no suporta directament l'ús de fonts OpenType i característiques tipogràfiques avançades.
2. **lualatex**: contràriament amb l'anterior, és una mica més lent a l'hora de compilar però suporta directament tipus de lletra TrueType i OpenType.

La plantilla LaTeX està dissenyada i configurada per compilar bé de les dues maneres. Vosaltres haureu d'escollir amb quina de les dues opcions us quedeu (podeu provar-les les dues per veure la diferència de velocitat). Personalment us recomanem la segona ja que és la més moderna i probablement serà la que acabarà prevalent a mig termini.

## Generació del document.

Coneixement:

El procés de generació d'un document LaTeX pot ser dividit en diversos passos. Aquí hi ha una breu explicació del procediment:

1. **Edició del document**: L'usuari escriu el contingut del document en un editor de text. El document està escrit en format LaTeX, que inclou comandes i marcadors per a indicar la seva estructura i format. Per ajudar en aquesta tasca hi ha editors de text especialment dedicats a això com ara **TeXStudio**.

2. **Compilació**: Per obtenir un document final (generalment en format PDF), cal compilar el document LaTeX. Aquest procés implica l'ús d'un motor LaTeX, com "pdflatex" o "lualatex", per processar el document i generar el resultat desitjat. Durant la compilació es generen diversos **fitxers auxiliars** que ajuden a LaTeX a gestionar la informació, com ara els índexs, les bibliografies i les referències creuades.

3. **Gestió de bibliografia**: Si s'utilitza una bibliografia, LaTeX utilitza un programari com "biber" per processar les cites i generar la llista de referències bibliogràfiques.

4. **Compilació múltiple**: En molts casos, és necessari compilar el document més d'una vegada per resoldre les referències creuades i altres elements que requereixen informació addicional.

5. **Generació del document final**: Finalment, el motor LaTeX crea el document final en el format desitjat, com un arxiu PDF. El document inclou tots els elements, com ara text, imatges, taules i gràfics, amb el format especificat pel document LaTeX original.

El resultat és un document de gran qualitat tipogràfica que segueix les normes estàndard de format i estil de LaTeX. Aquest procés de compilació permet als autors centrar-se en el contingut del document mentre LaTeX gestiona la presentació i la tipografia de manera automàtica.

Així, la primera vegada que compilem el document caldrà fer:

```sh
pdflatex TFG.tex (repetir l'ordre dues o més vegades per resoldre les referències creuades)
biber TFG        (això genera la bibliografia)
pdflatex TFG.tex (repetir 2 o més vegades per incorporar la bibliog. i resoldre les referències)
``` 

o:

```sh
lualatex TFG.tex (repetir l'ordre dues o més vegades per resoldre les referències creuades)
biber TFG        (això genera la bibliografia)
lualatex TFG.tex (repetir 2 o més vegades per incorporar la bibliog. i resoldre les referències)
```

Una vegada generat sense errors ni advertències, si no modifiquem la bibliografia podrem anar actualitzant únicament el fitxer .TeX principal amb `lualatex/pdflatex` i prou.

## Personalització de la plantilla (IMPORTANT !)

### Informació del TFG
Per adaptar la plantilla a les vostres necessitats heu d'**editar el fitxer principal (`TFG.tex`)** i introduir la vostra informació personal:

1. Indiqueu l'idioma del document a la variable `\doclanguage` (línia 31).
2. Descomenteu la macro `\setTitle` i poseu-hi el títol del vostre TFG (línia 37)
3. Feu el mateix amb les macros `\setAuthor`, `\setAdvisor`, `\setReviewer` i `\setDate`
4. Feu el mateix amb la macro `\setDegree`, però aquí no hi heu de posar un text qualsevol sinó que ja hi ha predefinides les variables \GRETST i \GREELEC que ja s'encarreguen de tot.
5. Canvieu el text dels diferents apartats d'agraïments, dedicatòria, etc.
6. Finalment, editeu el contingut dels diferents fitxers en què està divit el document per posar-hi el vostre propi contingut, escrit en l'idioma que hàgiu escollit.

### Canvis d'estil

Si voleu canviar o afegir alguna cosa en l'estil del document, la majoria d'opcions les trobareu al fitxer `config/styles.tex`. De tota manera com que es tracta d'un document oficial i formal no convé canviar l'aparença general.

### Afegir funcionalitat

Per ampliar les capacitats del LaTeX podeu afegir paquets al preàmbul. El lloc ideal per fer-ho és al fitxer `config/packages.tex`.