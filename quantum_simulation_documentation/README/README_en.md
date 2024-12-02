# LaTeX Template for Bachelor's Thesis

(send comments and/or suggestions to [orestes.mas@upc.edu](mailto:orestes.mas@upc.edu))

## Languages

This TFG template allows you to generate the PDF in 3 different languages: Catalan, Spanish, and English. To select the generation language, please refer to the "Customization" section.

## Requirements

To compile the document and obtain a final PDF, you will need a modern and functional TeX installation. It can be installed manually, but it's much easier to install a **precompiled TeX distribution**.

1. **For GNU/Linux systems**, the [TeX Live][1] distribution is the predominant one, although some years ago the [MiKTeX][3] distribution was also ported to GNU/Linux systems. Most GNU/Linux distributions, like Ubuntu, Mint, Debian, or Fedora, already have TeX Live packaged and ready to install, so check your package manager.
2. **For macOS computers**, you can install [MacTeX][2], which is essentially TeX Live for Mac.
3. **For Windows**, you can choose between [MiKTeX][3] or [TeX Live]. The former is more popular among Windows users, probably because it has been available for more years.
4. **Online**: Some people start their first steps in LaTeX using an online service like [Overleaf][4]. While these services can be helpful in lowering the entry barrier, keep in mind that they are _freemium_ services, meaning the free version has some limitations (usually in document length/processing time), and you need to upgrade to a paid version to remove those limitations.

[1]: https://tug.org/texlive/
[2]: https://tug.org/mactex/
[3]: https://miktex.org/
[4]: https://www.overleaf.com/

## Considerations

LaTeX is a very old program that has evolved over time. Additionally, older versions have been preserved to maintain compatibility. This means that there are currently 2 main ways to generate your document:

1. **pdflatex**: This is the "older" way, which compiles faster but does not directly support the use of OpenType fonts and advanced typographic features.
2. **lualatex**: In contrast to the former, it compiles a bit slower but directly supports TrueType and OpenType fonts.

The LaTeX template is designed and configured to compile well using both methods. You will need to choose which of the two options you prefer (you can try both to see the difference in speed). Personally, we recommend the second one as it is more modern and likely to prevail in the medium term.

## Document Generation

Knowledge:

The process of generating a LaTeX document can be divided into several steps. Here's a brief explanation of the procedure:

1. **Document Editing**: The user writes the content of the document in a text editor. The document is written in LaTeX format, which includes commands and markers to indicate its structure and formatting. To assist with this task, there are text editors specially dedicated to LaTeX, such as TeXStudio.

2. **Compilation**: To obtain a final document (usually in PDF format), you need to compile the LaTeX document. This process involves using a LaTeX engine, such as "pdflatex" or "lualatex," to process the document and generate the desired output. During compilation, several **auxiliary files** are generated to help LaTeX manage information, such as indexes, bibliographies, and cross-references.

3. **Bibliography Management**: If a bibliography is used, LaTeX uses software like "biber" to process citations and generate the list of bibliographic references.

4. **Multiple Compilation**: In many cases, it's necessary to compile the document more than once to resolve cross-references and other elements that require additional information.

5. **Generation of the Final Document**: Finally, the LaTeX engine creates the final document in the desired format, such as a PDF file. The document includes all elements, such as text, images, tables, and graphics, with the formatting specified in the original LaTeX document.

The result is a document with high typographic quality that follows standard formatting and style rules of LaTeX. This compilation process allows authors to focus on the content of the document while LaTeX automatically manages presentation and typography.

Therefore, the first time you compile the document, you will need to do the following:

```sh
pdflatex TFG.tex  (repeat this command 2 or 3 times to resolve the cross references)
biber TFG         (this command generates the bibliography)
pdflatex TFG.tex  (repeat this command 2 or 3 times to resolve the cross references)
``` 

or:

```sh
lualatex TFG.tex  (repeat this command 2 or 3 times to resolve the cross references)
biber TFG         (this command generates the bibliography)
lualatex TFG.tex  (repeat this command 2 or 3 times to resolve the cross references)
```

Once generated without errors or warnings, if you don't modify the bibliography, you can update only the main .TeX file with `lualatex/pdflatex` and that's it.

## Template Customization (IMPORTANT!)

### Bachelor's Thesis Information

To adapt the template to your needs, you need to **edit the main file (`TFG.tex`)** and enter your personal information:

1. Specify the document's language in the `\doclanguage` variable (line 31).
2. Uncomment the `\setTitle` macro and enter the title of your bachelor's thesis (line 37).
3. Do the same with the macros `\setAuthor`, `\setAdvisor`, `\setReviewer`, and `\setDate`.
4. Do the same with the `\setDegree` macro, but here you don't need to enter arbitrary text as predefined variables like `\GRETST` and `\GREELEC` take care of everything.
5. Change the text of different sections, such as acknowledgments, dedication, etc.
6. Finally, edit the content of the different files in which the document is divided to add your own content, written in the language you have chosen.

### Style Changes

If you want to change or add something to the document's style, most options can be found in the `config/styles.tex` file. However, since it is an official and formal document, it is advisable not to change the general appearance.

### Adding Functionality

To expand LaTeX's capabilities, you can add packages to the preamble. The ideal place to do this is in the `config/packages.tex` file.