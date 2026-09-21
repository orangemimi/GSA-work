import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "/Users/mimi/Documents/Code/Github/GSA-work/explore/sythetic/outputs/manual_branch_labels_20260830";
const previewDir = `${outputDir}/previews`;
await fs.mkdir(previewDir, { recursive: true });

const MODELS = ["CESM2", "CNRM-CM6-1", "CanESM5", "GFDL-CM4", "CMCC-CM2-SR5"];
const DOMAINS = ["all_land", "WW", "WD", "CW", "CD", "LI"];
const DOMAIN_HEADERS = ["All land", "Wet-warm (WW)", "Dry-warm (WD)", "Wet-cold (CW)", "Dry-cold (CD)", "Land ice (LI)"];
const STATUSES = ["Branch", "Candidate", "Uncertain", "Two-band", "No branch", "待标注"];
const COLORS = {
  "Branch": "#D9F2E3",
  "Candidate": "#FFF1B8",
  "Uncertain": "#E5E7EB",
  "Two-band": "#E8DDF7",
  "No branch": "#F5F6F8",
  "待标注": "#FFFFFF",
};
const FONT_COLORS = {
  "Branch": "#166534",
  "Candidate": "#8A5A00",
  "Uncertain": "#4B5563",
  "Two-band": "#6B21A8",
  "No branch": "#374151",
  "待标注": "#9CA3AF",
};

const sheets = [
  {
    name: "P_Q_TRIM",
    pair: "P → Q",
    version: "TRIMMED",
    labels: {
      "CESM2|all_land": "No branch", "CESM2|WW": "Two-band", "CESM2|WD": "No branch",
      "CESM2|CW": "No branch", "CESM2|CD": "No branch", "CESM2|LI": "Branch",
      "CNRM-CM6-1|all_land": "Uncertain", "CNRM-CM6-1|WW": "No branch", "CNRM-CM6-1|WD": "No branch",
      "CNRM-CM6-1|CW": "No branch", "CNRM-CM6-1|CD": "No branch", "CNRM-CM6-1|LI": "No branch",
      "CanESM5|all_land": "Uncertain", "CanESM5|WW": "No branch", "CanESM5|WD": "No branch",
      "CanESM5|CW": "No branch", "CanESM5|CD": "No branch", "CanESM5|LI": "No branch",
      "GFDL-CM4|all_land": "Uncertain", "GFDL-CM4|WW": "No branch", "GFDL-CM4|WD": "No branch",
      "GFDL-CM4|CW": "No branch", "GFDL-CM4|CD": "No branch", "GFDL-CM4|LI": "Branch",
      "CMCC-CM2-SR5|all_land": "No branch", "CMCC-CM2-SR5|WW": "No branch", "CMCC-CM2-SR5|WD": "No branch",
      "CMCC-CM2-SR5|CW": "No branch", "CMCC-CM2-SR5|CD": "No branch", "CMCC-CM2-SR5|LI": "Branch",
    },
    notes: {
      "CESM2|LI": "用户明确确认的 Branch（1,6）。",
      "GFDL-CM4|LI": "用户后期明确确认的 Branch（4,6）；旧参考表曾标 Uncertain。",
      "CMCC-CM2-SR5|LI": "用户明确确认的 Branch（5,6）。",
      "CESM2|WW": "旧人工参考记录为 Two-band。",
      "CESM2|CW": "用户明确认为应为 No branch。",
      "CNRM-CM6-1|all_land": "后期红框与早期判断存在冲突，保留 Uncertain。",
      "CanESM5|all_land": "后期红框与早期判断存在冲突，保留 Uncertain。",
      "GFDL-CM4|all_land": "后期红框与早期判断存在冲突，保留 Uncertain。",
    },
  },
  {
    name: "P_Q_RAW", pair: "P → Q", version: "RAW",
    labels: { "CanESM5|all_land": "No branch", "GFDL-CM4|LI": "Branch", "CMCC-CM2-SR5|LI": "Branch" },
    notes: {
      "CanESM5|all_land": "用户明确要求不要分成 Branch。",
      "GFDL-CM4|LI": "用户确认的 RAW Branch。",
      "CMCC-CM2-SR5|LI": "用户确认的 RAW Branch。",
    },
  },
  {
    name: "ET_Q_RAW", pair: "ET → Q", version: "RAW",
    labels: { "CESM2|CD": "No branch" },
    notes: { "CESM2|CD": "用户认为扇形/异方差结构不应作为 Branch。" },
  },
  {
    name: "RSDS_Q_RAW", pair: "rsds → Q", version: "RAW",
    labels: { "CESM2|WW": "No branch" },
    notes: { "CESM2|WW": "用户明确认为是误报。" },
  },
  {
    name: "RLDS_Q_RAW", pair: "rlds → Q", version: "RAW",
    labels: { "CESM2|WW": "No branch" },
    notes: { "CESM2|WW": "用户要求在不增加阈值的情况下筛掉。" },
  },
  {
    name: "HFLS_Q_RAW", pair: "hfls → Q", version: "RAW",
    labels: { "CESM2|all_land": "No branch" },
    notes: { "CESM2|all_land": "用户认为宽带/扇形结构不是 Branch。" },
  },
  {
    name: "HFSS_Q_RAW", pair: "hfss → Q", version: "RAW",
    labels: { "CESM2|WD": "No branch" },
    notes: { "CESM2|WD": "用户红框指出不应为 Branch。" },
  },
  {
    name: "HFSS_Q_TRIM", pair: "hfss → Q", version: "TRIMMED",
    labels: { "CESM2|WD": "No branch", "CESM2|LI": "No branch", "GFDL-CM4|LI": "No branch" },
    notes: {
      "CESM2|WD": "用户红框指出不应为 Branch。",
      "CESM2|LI": "用户红框指出不应为 Branch。",
      "GFDL-CM4|LI": "同组截图中被作为误报讨论。",
    },
  },
  {
    name: "P_ET_RAW", pair: "P → ET", version: "RAW",
    labels: { "CESM2|LI": "No branch", "CMCC-CM2-SR5|LI": "No branch" },
    notes: {
      "CESM2|LI": "用户红框指出不应为 Branch。",
      "CMCC-CM2-SR5|LI": "用户红框指出不应为 Branch。",
    },
  },
  {
    name: "MRSOS_Q_TRIM", pair: "mrsos → Q", version: "TRIMMED",
    labels: { "CMCC-CM2-SR5|LI": "No branch" },
    notes: { "CMCC-CM2-SR5|LI": "用户认为该面板不应为 Branch。" },
  },
  {
    name: "RLUS_Q_RAW", pair: "rlus → Q", version: "RAW",
    labels: {
      "CESM2|LI": "No branch", "GFDL-CM4|all_land": "No branch", "GFDL-CM4|WD": "No branch",
      "GFDL-CM4|LI": "No branch", "CMCC-CM2-SR5|WW": "No branch",
    },
    notes: {
      "CESM2|LI": "用户明确表示该组候选全部不是 Branch。",
      "GFDL-CM4|all_land": "用户明确表示该组候选全部不是 Branch。",
      "GFDL-CM4|WD": "用户明确表示该组候选全部不是 Branch。",
      "GFDL-CM4|LI": "用户明确表示该组候选全部不是 Branch。",
      "CMCC-CM2-SR5|WW": "用户明确表示该组候选全部不是 Branch。",
    },
  },
  {
    name: "RLUS_Q_TRIM", pair: "rlus → Q", version: "TRIMMED",
    labels: {
      "CESM2|all_land": "No branch", "CESM2|LI": "No branch",
      "GFDL-CM4|all_land": "No branch", "CMCC-CM2-SR5|WW": "No branch",
    },
    notes: {
      "CESM2|all_land": "用户明确表示该组候选全部不是 Branch。",
      "CESM2|LI": "用户明确表示该组候选全部不是 Branch。",
      "GFDL-CM4|all_land": "用户明确表示该组候选全部不是 Branch。",
      "CMCC-CM2-SR5|WW": "用户明确表示该组候选全部不是 Branch。",
    },
  },
  {
    name: "MRROS_Q_TRIM", pair: "mrros → Q", version: "TRIMMED",
    labels: { "CESM2|WD": "Uncertain", "CESM2|LI": "Uncertain", "CanESM5|CW": "Candidate" },
    notes: {
      "CESM2|WD": "用户询问为什么不是 Branch；先保留为 Uncertain。",
      "CESM2|LI": "用户认为也可能是 Branch；尚未最终确认。",
      "CanESM5|CW": "用户明确认为是局部分支，标为 Candidate。",
    },
  },
  {
    name: "HFLS_Q_TRIM", pair: "hfls → Q", version: "TRIMMED",
    labels: { "CESM2|LI": "Candidate", "GFDL-CM4|LI": "Candidate" },
    notes: {
      "CESM2|LI": "用户希望这类小分支能够被接受；暂列 Candidate。",
      "GFDL-CM4|LI": "用户希望这类小分支能够被接受；暂列 Candidate。",
    },
  },
  {
    name: "PRSN_Q_TRIM", pair: "prsn → Q", version: "TRIMMED",
    labels: {
      "CESM2|all_land": "Uncertain", "GFDL-CM4|all_land": "Uncertain",
      "GFDL-CM4|LI": "Uncertain", "CMCC-CM2-SR5|LI": "Uncertain",
    },
    notes: {
      "CESM2|all_land": "用户红框指出可能是漏检 Branch，尚未最终确认。",
      "GFDL-CM4|all_land": "用户红框指出可能是漏检 Branch，尚未最终确认。",
      "GFDL-CM4|LI": "用户红框指出可能是漏检 Branch，尚未最终确认。",
      "CMCC-CM2-SR5|LI": "用户红框指出可能是漏检 Branch，尚未最终确认。",
    },
  },
];

const workbook = Workbook.create();

function setTitle(sheet, title, subtitle, endCol = "G") {
  sheet.mergeCells(`A1:${endCol}1`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${endCol}1`).format = {
    fill: "#16324F", font: { bold: true, color: "#FFFFFF", size: 16 },
    horizontalAlignment: "center", verticalAlignment: "center",
  };
  sheet.getRange(`A1:${endCol}1`).format.rowHeightPx = 34;
  sheet.mergeCells(`A2:${endCol}2`);
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${endCol}2`).format = {
    fill: "#EAF0F6", font: { color: "#334155", italic: true, size: 10 },
    horizontalAlignment: "left", verticalAlignment: "center", wrapText: true,
  };
  sheet.getRange(`A2:${endCol}2`).format.rowHeightPx = 30;
}

function styleStatusGrid(range) {
  range.format = {
    font: { size: 10, color: "#374151" }, horizontalAlignment: "center",
    verticalAlignment: "center", wrapText: true,
    borders: { preset: "all", style: "thin", color: "#D7DEE7" },
  };
  for (const status of STATUSES) {
    const safe = status.replaceAll('"', '""');
    range.conditionalFormats.addCustom(`=B5="${safe}"`, {
      fill: COLORS[status], font: { bold: status !== "待标注", color: FONT_COLORS[status] },
    });
  }
}

// Summary sheet
const summary = workbook.worksheets.add("Summary");
summary.showGridLines = false;
setTitle(summary, "CMIP6 Branch Manual Label Workbook", "人工判断优先；算法输出 S:/K: 不自动写入人工标签。所有矩阵均为 5 个模型 × 6 个区域。", "J");
summary.getRange("A4:B10").values = [
  ["标签", "含义"],
  ["Branch", "明确分支"],
  ["Candidate", "局部分支或较弱证据"],
  ["Uncertain", "存在冲突或尚未最终确认"],
  ["Two-band", "持续双带，但不一定展开成分叉"],
  ["No branch", "明确不是分支"],
  ["待标注", "尚无人工判断"],
];
summary.getRange("A4:B4").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" } };
for (let i = 0; i < STATUSES.length; i++) {
  summary.getRange(`A${5 + i}:B${5 + i}`).format = {
    fill: COLORS[STATUSES[i]], font: { color: FONT_COLORS[STATUSES[i]], bold: true },
    borders: { preset: "inside", style: "thin", color: "#D7DEE7" },
  };
}
summary.getRange("A12:J12").values = [["Worksheet", "Pair", "Version", "Annotated", "Branch", "Candidate", "Uncertain", "Two-band", "No branch", "待标注"]];
summary.getRange("A12:J12").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
const summaryValues = sheets.map((cfg) => [cfg.name, cfg.pair, cfg.version, null, null, null, null, null, null, null]);
summary.getRange(`A13:J${12 + sheets.length}`).values = summaryValues;
for (let i = 0; i < sheets.length; i++) {
  const row = 13 + i;
  const sheetName = sheets[i].name;
  summary.getRange(`D${row}:J${row}`).formulas = [[
    `=30-COUNTIF('${sheetName}'!B5:G9,"待标注")`,
    `=COUNTIF('${sheetName}'!B5:G9,"Branch")`,
    `=COUNTIF('${sheetName}'!B5:G9,"Candidate")`,
    `=COUNTIF('${sheetName}'!B5:G9,"Uncertain")`,
    `=COUNTIF('${sheetName}'!B5:G9,"Two-band")`,
    `=COUNTIF('${sheetName}'!B5:G9,"No branch")`,
    `=COUNTIF('${sheetName}'!B5:G9,"待标注")`,
  ]];
}
summary.getRange(`A13:J${12 + sheets.length}`).format = {
  borders: { preset: "inside", style: "thin", color: "#E1E6EC" },
  verticalAlignment: "center",
};
summary.getRange(`D13:J${12 + sheets.length}`).format.horizontalAlignment = "center";
summary.getRange("A4:A10").format.columnWidthPx = 105;
summary.getRange("B4:B10").format.columnWidthPx = 260;
summary.getRange("A12:A27").format.columnWidthPx = 125;
summary.getRange("B12:B27").format.columnWidthPx = 110;
summary.getRange("C12:C27").format.columnWidthPx = 90;
summary.getRange("D12:J27").format.columnWidthPx = 78;
summary.freezePanes.freezeRows(12);

// Matrix sheets
for (const cfg of sheets) {
  const sheet = workbook.worksheets.add(cfg.name);
  sheet.showGridLines = false;
  setTitle(sheet, `${cfg.pair} · ${cfg.version} · Manual labels`, "可直接修改 B5:G9；下拉菜单限定为六类标签。颜色会随文字标签自动变化。", "J");
  sheet.getRange("A4:G4").values = [["Model", ...DOMAIN_HEADERS]];
  sheet.getRange("A4:G4").format = {
    fill: "#1F4E78", font: { bold: true, color: "#FFFFFF", size: 10 },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
    borders: { preset: "all", style: "thin", color: "#D7DEE7" },
  };
  sheet.getRange("A5:A9").values = MODELS.map((model) => [model]);
  sheet.getRange("A5:A9").format = {
    fill: "#EAF0F6", font: { bold: true, color: "#1F2937" },
    verticalAlignment: "center", borders: { preset: "all", style: "thin", color: "#D7DEE7" },
  };
  const grid = MODELS.map((model) => DOMAINS.map((domain) => cfg.labels[`${model}|${domain}`] ?? "待标注"));
  sheet.getRange("B5:G9").values = grid;
  sheet.getRange("B5:G9").dataValidation = { rule: { type: "list", values: STATUSES } };
  styleStatusGrid(sheet.getRange("B5:G9"));
  sheet.getRange("A4:A9").format.columnWidthPx = 142;
  sheet.getRange("B4:G9").format.columnWidthPx = 116;
  sheet.getRange("A4:G4").format.rowHeightPx = 42;
  sheet.getRange("A5:G9").format.rowHeightPx = 34;

  sheet.getRange("I4:J10").values = [
    ["Legend", "Meaning"],
    ["Branch", "明确分支"],
    ["Candidate", "局部/较弱分支"],
    ["Uncertain", "尚需复核"],
    ["Two-band", "双带"],
    ["No branch", "明确非分支"],
    ["待标注", "无人工判断"],
  ];
  sheet.getRange("I4:J4").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" } };
  for (let i = 0; i < STATUSES.length; i++) {
    sheet.getRange(`I${5 + i}:J${5 + i}`).format = {
      fill: COLORS[STATUSES[i]], font: { bold: true, color: FONT_COLORS[STATUSES[i]] },
      borders: { preset: "inside", style: "thin", color: "#D7DEE7" },
    };
  }
  sheet.getRange("I4:I10").format.columnWidthPx = 90;
  sheet.getRange("J4:J10").format.columnWidthPx = 130;
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(1);
}

// Flat, machine-readable label log
const master = workbook.worksheets.add("Label_Log");
master.showGridLines = false;
setTitle(master, "Manual Label Log", "每行是一条已有人工作出判断的记录；矩阵中的“待标注”不写入此表。Confidence 表示本次整理时的确定程度。", "H");
master.getRange("A4:H4").values = [["Pair", "Version", "Model", "Domain", "Coordinate", "Manual label", "Confidence", "Basis / note"]];
master.getRange("A4:H4").format = {
  fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
};
const logRows = [];
for (const cfg of sheets) {
  for (let r = 0; r < MODELS.length; r++) {
    for (let c = 0; c < DOMAINS.length; c++) {
      const key = `${MODELS[r]}|${DOMAINS[c]}`;
      const label = cfg.labels[key];
      if (!label) continue;
      let confidence = "Medium";
      if (label === "Uncertain") confidence = "Low";
      else if (cfg.notes[key]?.includes("明确") || cfg.notes[key]?.includes("确认")) confidence = "High";
      logRows.push([cfg.pair, cfg.version, MODELS[r], DOMAINS[c], `(${r + 1},${c + 1})`, label, confidence, cfg.notes[key] ?? "整理自既有人工参考。"]);
    }
  }
}
master.getRange(`A5:H${4 + logRows.length}`).values = logRows;
master.getRange(`A5:H${4 + logRows.length}`).format = {
  borders: { preset: "inside", style: "thin", color: "#E1E6EC" },
  verticalAlignment: "top", wrapText: true,
};
master.getRange(`F5:F${4 + logRows.length}`).dataValidation = { rule: { type: "list", values: STATUSES.slice(0, 5) } };
for (const status of STATUSES.slice(0, 5)) {
  master.getRange(`F5:F${4 + logRows.length}`).conditionalFormats.addCustom(`=F5="${status}"`, {
    fill: COLORS[status], font: { bold: true, color: FONT_COLORS[status] },
  });
}
master.getRange("A4:A100").format.columnWidthPx = 92;
master.getRange("B4:B100").format.columnWidthPx = 84;
master.getRange("C4:C100").format.columnWidthPx = 140;
master.getRange("D4:D100").format.columnWidthPx = 90;
master.getRange("E4:E100").format.columnWidthPx = 82;
master.getRange("F4:F100").format.columnWidthPx = 102;
master.getRange("G4:G100").format.columnWidthPx = 84;
master.getRange("H4:H100").format.columnWidthPx = 330;
master.freezePanes.freezeRows(4);

// Compact verification before export.
console.log((await workbook.inspect({
  kind: "table", range: "Summary!A12:J27", include: "values,formulas",
  tableMaxRows: 20, tableMaxCols: 12, maxChars: 8000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan",
})).ndjson);

for (const name of ["Summary", ...sheets.map((s) => s.name), "Label_Log"]) {
  const range = name === "Summary" ? "A1:J27" : name === "Label_Log" ? `A1:H${4 + logRows.length}` : "A1:J10";
  const preview = await workbook.render({ sheetName: name, range, scale: 1.25, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${outputDir}/CMIP6_branch_manual_labels.xlsx`);
console.log(`Saved ${outputDir}/CMIP6_branch_manual_labels.xlsx`);
console.log(`Log rows: ${logRows.length}`);
