# GEAS DEMO - Mapping (instancia inversión 8792)

Estos tickets DEMO muestran un camino realista: ONNX + sentimiento + calidad de señales + gestión de riesgo de cartera + drift/monitor + ensemble + backtest realista.

## Tickets GEAS (FREE)

- DEMO-001: c812d8ae-aa19-497f-bb7b-a7bd31db51e1
- DEMO-002: da378aa5-4bec-469f-8943-4cb20f9f78e3
- DEMO-003: d44944f6-448e-4821-ac9c-674670542843
- DEMO-004: 512446f1-70b7-443c-b786-006878030309
- DEMO-005: 8ae253bb-7627-474a-ac88-75fc923d2d4d
- DEMO-006: a9ba3f68-ca0f-4024-af3f-37b0ad13f579
- DEMO-007: dbe501ac-1568-460a-bb12-6195327b0adc
- DEMO-008: 34597f25-3ecf-4c7b-b83f-7a1f6ead0a4c
- DEMO-009: f6bd3f02-5acb-449d-a7c0-0560528473ae
- DEMO-010: 54fd791d-cd8d-4c5c-87ac-49be0464381f

## Secuencia (para empezar desde portatil)
1. DEMO-001 ONNX export + inferencia estable
2. DEMO-002 Sentimiento financiero (VADER+reglas) -> integrar a features
3. DEMO-003 SHAP + Conformal + umbral confianza
4. DEMO-004 Gestión riesgo cartera (Markowitz) + sizing Kelly fraccional
5. DEMO-005 Monitoring drift + alertas + backtest realista
6. DEMO-006 Ensemble híbrido RF+HistGBM con validación honesta
7. DEMO-007 Inferencia ONNX CLI/API + benchmark
8. DEMO-008 Sentimiento real NewsAPI.ai (vivo) + agregación diaria
9. DEMO-009 Conformal gate + umbral dinámico por volatilidad
10. DEMO-010 Backtest realista completo (costes/slippage/drawdown)

## Uso desde portátil
```bash
export GEAS_URL=http://192.168.1.145:8792 GEAS_TOKEN=taHBDs6M0MIrmqlOW4wVh6KPsLrM1M7x GEAS_ACTOR=Portatil
cd ~/Stock-Market-Prediction
./start.sh <DEMO-XXX-GEAS-ID>
./status.sh --json
./finish.sh <DEMO-XXX-GEAS-ID>
```

Panel: http://192.168.1.145:8792/

## Sin exportar cada vez (portátil)

Opción A: usar script (source)
```bash
cd ~/Stock-Market-Prediction
source scripts/geas-remote.sh
```

Opción B: añadir al .bashrc/.zshrc (permanente)
```bash
echo 'export GEAS_URL=http://192.168.1.145:8792 GEAS_TOKEN=taHBDs6M0MIrmqlOW4wVh6KPsLrM1M7x GEAS_ACTOR=Portatil' >> ~/.bashrc
source ~/.bashrc
```

Opción C: usar direnv (.envrc)
```bash
cp .envrc.example .envrc
direnv allow
```

## Tickets PORT (copiar ideas/código útil desde Edgebet)

- PORT-001: a68848d1-65fc-4ef4-b408-271a0a2eb197
- PORT-002: a9dd46fa-57d3-403e-8231-1566d81318b2
- PORT-003: 95f58f7c-3dfe-4ec4-a730-a94b8553eb41
- PORT-004: 579feb89-7890-4c1c-9bd3-f7db761e8d97
- PORT-005: a001fd89-9e9d-44d1-983a-bb35da777687
