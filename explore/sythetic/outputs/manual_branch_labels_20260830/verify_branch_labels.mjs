import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const path = "/Users/mimi/Documents/Code/Github/GSA-work/explore/sythetic/outputs/manual_branch_labels_20260830/CMIP6_branch_manual_labels.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));

console.log((await workbook.inspect({
  kind: "table", range: "P_Q_TRIM!A4:G9", include: "values,formulas",
  tableMaxRows: 10, tableMaxCols: 8, maxChars: 5000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "table", range: "MRROS_Q_TRIM!A4:G9", include: "values,formulas",
  tableMaxRows: 10, tableMaxCols: 8, maxChars: 5000,
})).ndjson);
console.log((await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 }, summary: "post-export formula error scan",
})).ndjson);
