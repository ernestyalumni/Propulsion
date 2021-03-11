------------------------------- MODULE Euclid -------------------------------
(* this specification extend the Integers standard module to get the
definitions of arithmetic operators (+,-,etc) *)
EXTENDS Integers

(* "p divides q" predicate iff there exists some integer d in interval 1..q
such that q is equal to p times d *)
p | q == \E d \in 1..q : q = p * d

(* Define set of divisors of an integer q as sets of integers which both
belong to the interval 1..q and divide q *)
Divisors(q) == {d \in 1..q : d | q}

(*  *)
Maximum(S) == CHOOSE x \in S : \A y \in S : x >= y


=============================================================================
\* Modification History
\* Last modified Wed Feb 03 11:20:01 PST 2021 by topolo
\* Created Wed Feb 03 10:20:39 PST 2021 by topolo
