ARCHITECTURE CODING GUIDELINES
Szigoru architekturalis fegyelmi szabalyzat
1. CEL

Ez a dokumentum az architektura vedelmet szolgalja.

A cel:

A modular monolith struktura megorzese

A retegek kozti hatarok betartasa

A globalis allapot megszuntetese

A jovobeni SaaS es trading bot irany tamogatasa

A rendszer hosszu tavu karbantarthatosaganak biztositasa

Ez NEM ajanlas.
Ez kotelezo ervenyu szabalyrendszer.

2. ARCHITEKTURAI ALAPELV

A rendszer 5 retegbol all:

interfaces
application
core
infrastructure
config/bootstrap


A retegek kozti fuggesi szabaly szigoruan egyiranyu.

3. FUGGESI SZABALYOK (KOTELEZO)
3.1 Engedelyezett fuggesek
interfaces -> application
application -> core
application -> infrastructure
bootstrap -> infrastructure
bootstrap -> application

3.2 TILTOTT fuggesek
Core retegben TILOS:

SQLite import

Flask import

Config import

Email import

OS env olvasas

Logging side-effect

Repository implementacio import

A core reteg:

100% tiszta domain logika

Application retegben TILOS:

Flask import

kozvetlen sqlite3 hasznalat

globalis Config import

Infrastructure retegben TILOS:

Core logikat modositani

uzleti donteseket hozni

Interfaces retegben TILOS:

SQL lekerdezes

Core logika implementacio

Repository peldanyositas

4. CONFIG SZABALYOK
4.1 Globalis Config hasznalata TILOS

Nem megengedett:

from app.config.config import Config


Minden konfiguracio:

Settings dataclass

konstruktor parameterkent kerul atadasra

immutable (frozen=True)

4.2 Env olvasas kizarolag bootstrapban

Csak:

app/config/build_settings.py


olvashat kornyezeti valtozot.

5. REPOSITORY SZABALYOK
5.1 Repository interfesz a core-ban

Core retegben csak:

Protocol / Interface


letezhet.

Implementacio kizarolag infrastructure-ban.

5.2 Kozvetlen SQLite hasznalat TILOS

Nem megengedett:

import sqlite3


a kovetkezo helyeken:

core

application

interfaces

6. FAJLMERET KORLATOK

Kotelezo limit:

Max 400 sor / fajl

Max 10 belso import / fajl

Max 1 felelosseg / modul

Ha egy fajl > 500 sor:

Refaktor kotelezo.

7. SERVICE KONSTRUKTOR SZABALY

Minden application service:

explicit dependency injection

nincs globalis hivatkozas

nincs implicit singleton

Pelda (helyes):

class DailyPipelineUseCase:
    def __init__(
        self,
        ohlcv_repo: IOhlcvRepository,
        decision_repo: IDecisionRepository,
        settings: Settings,
    ):
        ...

8. TILOS PATTERNOK
8.1 God Object

Egy osztaly:

nem kezelhet adatbazist

nem tartalmazhat uzleti logikat

nem vegezhet migraciot

nem lehet 1000+ sor

8.2 Service Locator

Tilos:

get_global_instance()

8.3 Import ciklus workaround

Lazy import NEM megoldas.

Ha circular dependency van:

Architektura serult.

8.4 Side-effect import

Tilos:

fajlrendszer muvelet importkor

DB inicializalas importkor

env validacio importkor

9. CORE RETEG SZABALYAI

Core:

determinisztikus

side-effect mentes

input -> output

nem modosit globalis allapotot

nem ir fajlba

nem logol kozvetlenul

Core csak:

adatot kap

adatot ad vissza

10. TESTELESI SZABALY

Core tesztek:

mock nelkuli unit teszt

Application tesztek:

repository mock

Infrastructure tesztek:

integration teszt

Ha egy core teszt mock-ot igenyel:

Architektura hibas.

11. BOOTSTRAP SZABALY

Csak bootstrapban:

dependency wiring

repository peldanyositas

use case peldanyositas

Bootstrap az egyetlen composition root.

12. LOGOLAS SZABALY

Core nem logol.

Application csak interfeszen keresztul logol.

Infrastructure kezeli a logger implementaciot.

13. RL / HEAVY TASK SZABALY

RL training:

kulon use case

nem hivhato automatikusan importkor

nem blokkolhat web requestet

14. RASPBERRY 5 SPECIFIKUS SZABALYOK

Egy process model

SQLite WAL mode

Nincs multiprocessing alapertelmezetten

Memory footprint minimalizalas

15. CODE REVIEW CHECKLIST (KOTELEZO)

Minden PR elott ellenorizni:

 Core nem importal infrastructure-t

 Nincs globalis Config

 Nincs sqlite3 import core-ban

 File < 400 sor

 Explicit dependency injection

 Nincs circular import

 Tesztek futnak

16. ARCHITEKTURA SERTES DEFINICIO

Architektura sertesnek minosul:

Globalis Config visszahozasa

Repository megkerulese

Core retegbe DB vagy Flask import

800+ soros service fajl

Lazy import workaround circular dependency-re

Ilyen esetben:

Refaktor kotelezo, nem opcionalis.

17. HOSSZU TAVU VIZIO

Ez a szabalyzat biztositja, hogy:

Raspberry 5 stabil marad

Trading bot irany nyitva marad

SaaS lehetoseg megmarad

Research rugalmassag megmarad

Technikai adossag kontroll alatt marad

18. ZARO ELV

Az architektura fontosabb, mint a feature.

Ha valasztani kell:

Stabil struktura > Gyors funkcio