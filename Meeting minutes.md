# Cornhole IoT

## 30-11-2022

Teilnehmer: Gabsteiger Florian, Florian Gritsch

### Aktueller Stand

* Gibt jetzt eine Scoreberechnung
* Gibt jetzt ein Github Repo: https://github.com/Steckdose007/Cornhole
* Notebook wurde in einzelne dokumente und Klassen aufgeteilt
* Netzt konnte noch nicht trainiert werden weil google colab den zugang speert und man neues volumen kaufen müsste
  - also wurde ein neuer PC gekauft
  - noch nicht alle Teile da
* Score ist vom Threshold des Netztes abhängig. manche modele brauchen großen manche kleinen.
### Nächste Schritte
* Weiterarbeit an der Netzwerkarchitektur
* Christian: Informationen über *nbstripout* bereitstellen
* Farberkennung verbessern im Orginalbild -> evt. eine art classifier der um die Prediction rum nach den Farben sucht
* google coral AI board
* Logistic Regression um anzahl säckchen zu bekommen.
* Farbraum verbessern
* Loss funktionen evaluieren (grid search)
* länge der prediction list egal? Weil wenn da weder rot noch blau ist dann kann da ja kein säckchen sein. Also vl threshold eher etwas gernger setzten. Also erstmal die Farberkennung hinbekommen.
