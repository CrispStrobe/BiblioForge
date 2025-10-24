#!/bin/bash
# Encoding: UTF-8

# Moving files for: anticartesianisme_Malebranche.pdf
mkdir -p /Users/christianstrobele/code/BiblioForge/UnknownAuthor

# Move original file
if [ -f /Users/christianstrobele/Documents/texte/anticartesianisme_Malebranche.pdf ]; then
  mv /Users/christianstrobele/Documents/texte/anticartesianisme_Malebranche.pdf '/Users/christianstrobele/code/BiblioForge/UnknownAuthor/1916 Lanticartésianisme de Malebranche.pdf'
  echo 'Moved: anticartesianisme_Malebranche.pdf -> /Users/christianstrobele/code/BiblioForge/UnknownAuthor/1916 Lanticartésianisme de Malebranche.pdf'
else
  echo 'Error: Source file /Users/christianstrobele/Documents/texte/anticartesianisme_Malebranche.pdf not found' >&2
fi

# Move associated text file
if [ -f /Users/christianstrobele/code/BiblioForge/anticartesianisme_Malebranche.txt ]; then
  mv /Users/christianstrobele/code/BiblioForge/anticartesianisme_Malebranche.txt '/Users/christianstrobele/code/BiblioForge/UnknownAuthor/1916 Lanticartésianisme de Malebranche.txt'
  echo 'Moved: anticartesianisme_Malebranche.txt -> /Users/christianstrobele/code/BiblioForge/UnknownAuthor/1916 Lanticartésianisme de Malebranche.txt'
else
  echo 'Info: Associated text file /Users/christianstrobele/code/BiblioForge/anticartesianisme_Malebranche.txt not found for move (original: /Users/christianstrobele/Documents/texte/anticartesianisme_Malebranche.pdf)' >&2
fi

# ================================

# Moving files for: association_inseparable.pdf
mkdir -p /Users/christianstrobele/code/BiblioForge/UnknownAuthor

# Move original file
if [ -f /Users/christianstrobele/Documents/texte/association_inseparable.pdf ]; then
  mv /Users/christianstrobele/Documents/texte/association_inseparable.pdf '/Users/christianstrobele/code/BiblioForge/UnknownAuthor/1888 Une association inséparable Lagrandissement des astres à lhorizon.pdf'
  echo 'Moved: association_inseparable.pdf -> /Users/christianstrobele/code/BiblioForge/UnknownAuthor/1888 Une association inséparable Lagrandissement des astres à lhorizon.pdf'
else
  echo 'Error: Source file /Users/christianstrobele/Documents/texte/association_inseparable.pdf not found' >&2
fi

# Move associated text file
if [ -f /Users/christianstrobele/code/BiblioForge/association_inseparable.txt ]; then
  mv /Users/christianstrobele/code/BiblioForge/association_inseparable.txt '/Users/christianstrobele/code/BiblioForge/UnknownAuthor/1888 Une association inséparable Lagrandissement des astres à lhorizon.txt'
  echo 'Moved: association_inseparable.txt -> /Users/christianstrobele/code/BiblioForge/UnknownAuthor/1888 Une association inséparable Lagrandissement des astres à lhorizon.txt'
else
  echo 'Info: Associated text file /Users/christianstrobele/code/BiblioForge/association_inseparable.txt not found for move (original: /Users/christianstrobele/Documents/texte/association_inseparable.pdf)' >&2
fi

# ================================

