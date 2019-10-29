**Canny**
- am calculat gradientii imaginii, pe x si pe y, folosind 2 convolutii cu cele 2 filtre Sobel
- am aplicat non-maximum suppression pentru a reduce dimensiunea muchiilor si pentru a pastra doar detaliile
relevante din imagine
- am aplicat thresholding pentru a accepta / rejecta pixelii, in functie de cele 2 nivele de prag alese
- am aplicat un algoritm simplu de edge tracking, al carui scop este sa reconsidere muchiile weak (ca fiind strong), daca au cel putin un vecin strong.

**Hough** (vectorized voting algorithm)
- tinand cont ca un cerc este definit de 3 parametri (centru\_x, centru\_y, raza),
este nevoie de un acumulator 3D
- astfel, in functie de intervalul ales pentru raza, se voteaza pentru fiecare pixel din imagine.
- ne intereseaza ce tuplu (centru\_x, centru\_y, raza) a obtinut cele mai multe voturi, deoarece acest tuplu
va reprezenta (cu un grad ridicat de certitudine) un cerc, in imaginea data ca parametru.
- am simplificat putin implementarea, in sensul in care functia hough(.) primeste ca parametru si numarul (n)
de cercuri dupa care cautarea se termina (astfel, aleg primele n tupluri cu numar maxim de voturi, care, foarte important, reprezinta n cercuri diferite - primele n tupluri din acumulator pot reprezenta, de exemplu, acelasi cerc, cu valori foarte putin modificate)

Pentru recolorarea imaginii, am ales sa folosesc 2 functii de biblioteca, pentru a-mi usura implementarea.
Astfel, algoritmul de recolorare este urmatorul:
- input: image.shape, gradients
- output: recolored\_image
- folosind informatiile date de gradientul imaginii, construiesc o matrice in care stochez, pentru fiecare pixel,
id-ul ariei / regiunii din care face parte - aplicand functia label din scipy.measure
- daca gradientul este 0, atunci pixelul respectiv nu este muchie (este fundal / regiune interioara)
- astfel, fiecare regiune va fi identificata unic, si pe baza acestei identificari se va face colorarea
- pentru a _completa_ pixelii care sunt _inconjurati_, aplic functia maximum\_filter (din scipy.ndimage)
- rezultatul este un fel de flood fill
- pentru a delimita regiunile, sortez descrescator matricea; asadar, obtin perechi (id\_regiune, dimensiune\_regiune)
- tinand cont de faptul ca in enunt se precizeaza ca putem _exploata_ simplitatea imaginii, considerand dimensiunile fundalului, a bradului si a trunchiului mai mari decat ale decoratiilor, rezulta ca avem urmatoarele
regiuni (in ordine descrescatoare a dimensiunii): fundal, brad, trunchi, [decoratii]
