----------------------------- MODULE semaphore -----------------------------
\* https://youtu.be/D_sh1nnX3zY  Giacomo Citi
\* declares a variable.
VARIABLE color

\* first formula with expected values.
\* It's a tautology, logical consequence.
TypeOK == color \in {"red", "green", "yellow"}

\* Try this by uncommenting out to try an error.
\* TypeOK == color \in {"red", "yellow"}

\* Temporal logic formula example.
AlwaysTypeOK == [] TypeOK

Init == color = "red"

\* Boilerplate values as an example to make it work. Uncomment out for your
\* first try.
\* TurnGreen == TRUE
\* TurnYellow == TRUE
\* TurnRed == TRUE

TurnGreen ==
    color = "red" /\
    color' = "green"

TurnYellow ==
    color = "green" /\
    color' = "yellow"

TurnRed ==
    color = "yellow" /\
    color' = "red"

Next == 
    TurnGreen \/
    TurnYellow \/
    TurnRed

\* Could write entire spec as temporal logic.
Spec == Init /\ [][Next]_color 

=============================================================================
\* Modification History
\* Last modified Wed Feb 03 22:28:58 PST 2021 by topolo
\* Created Wed Feb 03 21:27:24 PST 2021 by topolo
