---
source: neurovault
type: report
tags: [neurovault, report]
---
# Session report: before and after

**Organizer:** `claude-code:haiku`  
**Captured:** 6 notes from 6 episodes  
**Intent gate:** 100% of intended thoughts caught, 0 false captures

## Word error rate

| | WER vs. intended thought |
|---|---|
| Before (raw decoder output) | 21% |
| After (filed note) | 2% |

The organizer removed **90%** of the decoding errors.

## Thought by thought

| # | Before: decoder output | After: filed note | Type | Conf. |
|---|---|---|---|---|
| 1 | E se o viult sogerisse quas notas antigas eu deveria revisar toda toda semana | **[[Vault sugerindo notas antigas para revisão semanal]]**<br>E se o vault sugerisse quais notas antigas eu deveria revisar toda semana? | idea | 0.67 |
| 2 | Prciso ranovar o domínio do meu site de portfólio antes de sexta | **[[Renovar domínio do site de portfólio até sexta]]**<br>Preciso renovar o domínio do meu site de portfólio antes de sexta. | task | 0.86 |
| 3 | Decudi usar o Clude para a revisão semnal em vez de fazer ne mão | **[[Usar o Claude para a revisão semanal]]**<br>Decidi usar o Claude para a revisão semanal em vez de fazer na mão. | decision | 0.70 |
| 4 | Por que eu sempre tenho ideias melhores andando do que sentado no computador? | **[[Ideias melhores caminhando do que sentado]]**<br>Por que eu sempre tenho ideias melhores andando do que sentado no computador? | reflection | 0.97 |
| 5 | Hoje foi um dia lento mos terminar aquele projeto du automação valeu a peni | **[[Dia lento, mas projeto de automação concluído]]**<br>Hoje foi um dia lento, mas terminar aquele projeto de automação valeu a pena. | reflection | 0.81 |
| 6 | Lembrar de testar o o headset de EEG com a da playback ontes de compror um | **[[Testar headset de EEG antes de comprar]]**<br>Lembrar de testar o headset de EEG com a da playback antes de comprar um. | task | 0.70 |

## What was actually meant (simulation ground truth)

1. E se o vault sugerisse quais notas antigas eu deveria revisar toda semana
2. Preciso renovar o domínio do meu site de portfólio antes de sexta
3. Decidi usar o Claude para a revisão semanal em vez de fazer na mão
4. Por que eu sempre tenho ideias melhores andando do que sentado no computador?
5. Hoje foi um dia lento mas terminar aquele projeto de automação valeu a pena
6. Lembrar de testar o headset de EEG com a placa de playback antes de comprar um
