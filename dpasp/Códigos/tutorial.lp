1/3::prize(1);1/3::prize(2);1/3::prize(3).
door(1). door(2). door(3).
goat(X) :- door(X), not prize(X).
select(1).
true.
open(1);open(2);open(3) :- true.
:- prize(X),open(X).
:- open(X),select(X).
choice(1);choice(2);choice(3) :- true.

:- choice(X),open(X).
:- choice(X),select(X).

win(keep)  :-select(X),prize(X).
win(change) :-choice(X),prize(X).

#query(win(change)|open(2)).
#semantics maxent.