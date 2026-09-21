/**
 * 3 slides: CMIP6 historical catalog is not an aligned matrix.
 * Run: node cmip6_catalog_not_aligned.js
 */
const pptxgen = require("pptxgenjs");

const COLOR = {
  darkBg: "1B2420",
  cream: "F1EDE3",
  beige: "E8DFC8",
  card: "F5EFE2",
  ink: "1B2420",
  body: "3B3F3A",
  muted: "7A7F76",
  forest: "3B5D3A",
  terracotta: "C26A33",
  hairline: "C9C4B4",
  titleOnDark: "EFE9D9",
  subOnDark: "C9C4B4",
};

const FONT = { head: "Georgia", body: "Calibri", mono: "Consolas" };
const W = 13.333;
const H = 7.5;
const MX = 0.85;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "CMIP6 catalog is not aligned";
pres.author = "Case A";

function eyebrow(slide, label, page, onDark) {
  const c = onDark ? COLOR.subOnDark : COLOR.muted;
  slide.addText(label, {
    x: MX, y: 0.42, w: 9, h: 0.28,
    fontFace: FONT.body, fontSize: 11, color: c, charSpacing: 2.5, margin: 0,
  });
  slide.addText(page, {
    x: W - MX - 1.4, y: 0.42, w: 1.4, h: 0.28,
    fontFace: FONT.body, fontSize: 11, color: c, align: "right", margin: 0,
  });
}

// ---------- Slide 1 ----------
{
  const s = pres.addSlide();
  s.background = { color: COLOR.darkBg };
  eyebrow(s, "CMIP6 historical  ·  ESGF catalog", "01", true);
  s.addText("Not a filled 75 × 1103 table", {
    x: MX, y: 1.35, w: 11.5, h: 0.85,
    fontFace: FONT.head, fontSize: 36, color: COLOR.titleOnDark, margin: 0,
  });
  s.addText("75 models and 1,103 variables exist in the archive.\nThey do not form one aligned matrix.", {
    x: MX, y: 2.3, w: 11, h: 0.85,
    fontFace: FONT.body, fontSize: 18, color: COLOR.subOnDark, margin: 0,
  });

  const stats = [
    { n: "75", l: "models" },
    { n: "1,103", l: "variable IDs" },
    { n: "23%", l: "cells filled" },
    { n: "0", l: "vars on all 75" },
  ];
  stats.forEach((st, i) => {
    const x = MX + i * 2.95;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 3.55, w: 2.7, h: 2.35,
      fill: { color: "24302B" },
    });
    s.addText(st.n, {
      x, y: 3.8, w: 2.7, h: 1.15,
      fontFace: FONT.head, fontSize: 40, color: COLOR.titleOnDark,
      align: "center", margin: 0,
    });
    s.addText(st.l, {
      x: x + 0.12, y: 5.05, w: 2.46, h: 0.5,
      fontFace: FONT.body, fontSize: 14, color: COLOR.subOnDark,
      align: "center", margin: 0,
    });
  });
}

// ---------- Slide 2 ----------
{
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "What “not aligned” means", "02", false);
  s.addText("Same name. Different packaging.", {
    x: MX, y: 0.85, w: 11.5, h: 0.55,
    fontFace: FONT.head, fontSize: 28, color: COLOR.ink, margin: 0,
  });

  // left card — aligns
  s.addShape(pres.shapes.RECTANGLE, {
    x: MX, y: 1.7, w: 5.55, h: 4.85,
    fill: { color: "E4EDE0" },
  });
  s.addText("ALIGNED", {
    x: MX + 0.35, y: 1.95, w: 4.9, h: 0.28,
    fontFace: FONT.body, fontSize: 12, color: COLOR.forest, charSpacing: 2, margin: 0,
  });
  s.addText("Scientific definition", {
    x: MX + 0.35, y: 2.3, w: 4.9, h: 0.4,
    fontFace: FONT.head, fontSize: 22, color: COLOR.ink, margin: 0,
  });
  s.addText([
    { text: "pr  is precipitation", options: { breakLine: true } },
    { text: "evspsbl  is total ET", options: { breakLine: true } },
    { text: "mrro  is total runoff", options: { breakLine: true } },
    { text: "Units follow the CMIP data request", options: { breakLine: true } },
    { text: "(kg m⁻² s⁻¹ for water fluxes)", options: {} },
  ], {
    x: MX + 0.35, y: 2.9, w: 4.9, h: 3.2,
    fontFace: FONT.body, fontSize: 16, color: COLOR.body, paraSpaceAfter: 10, margin: 0,
  });

  // right card — does not
  s.addShape(pres.shapes.RECTANGLE, {
    x: MX + 5.85, y: 1.7, w: 5.75, h: 4.85,
    fill: { color: "F6E4D6" },
  });
  s.addText("NOT ALIGNED", {
    x: MX + 6.2, y: 1.95, w: 5.15, h: 0.28,
    fontFace: FONT.body, fontSize: 12, color: COLOR.terracotta, charSpacing: 2, margin: 0,
  });
  s.addText("Who has it, and how", {
    x: MX + 6.2, y: 2.3, w: 5.15, h: 0.4,
    fontFace: FONT.head, fontSize: 22, color: COLOR.ink, margin: 0,
  });
  s.addText([
    { text: "Coverage  —  a model has ~200 of 1,103 vars", options: { breakLine: true } },
    { text: "Table  —  Amon only, or also day / 3hr", options: { breakLine: true } },
    { text: "Grid  —  gn / gr / gr1", options: { breakLine: true } },
    { text: "Time  —  calendar, stop date, Flag vs Pass", options: { breakLine: true } },
    { text: "File attrs  —  fill value, member, version", options: {} },
  ], {
    x: MX + 6.2, y: 2.9, w: 5.15, h: 3.2,
    fontFace: FONT.body, fontSize: 16, color: COLOR.body, paraSpaceAfter: 10, margin: 0,
  });
}

// ---------- Slide 3 ----------
{
  const s = pres.addSlide();
  s.background = { color: COLOR.cream };
  eyebrow(s, "What we do instead", "03", false);
  s.addText("Cut an aligned subset. Do not force the rectangle.", {
    x: MX, y: 0.85, w: 11.6, h: 0.55,
    fontFace: FONT.head, fontSize: 26, color: COLOR.ink, margin: 0,
  });

  const steps = [
    { k: "01", t: "Long table", d: "One row = one model × variable.\nNot a 4,400-column matrix." },
    { k: "02", t: "Compatibility", d: "Same member, period, grid\nfor P, ET, R first." },
    { k: "03", t: "First batch", d: "11 models × 17 variables\n+ areacella and sftlf." },
  ];
  steps.forEach((st, i) => {
    const x = MX + i * 3.95;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.7, w: 3.7, h: 3.55,
      fill: { color: COLOR.beige },
    });
    s.addText(st.k, {
      x: x + 0.28, y: 1.95, w: 3.15, h: 0.4,
      fontFace: FONT.mono, fontSize: 14, color: COLOR.terracotta, margin: 0,
    });
    s.addText(st.t, {
      x: x + 0.28, y: 2.45, w: 3.15, h: 0.55,
      fontFace: FONT.head, fontSize: 22, color: COLOR.ink, margin: 0,
    });
    s.addText(st.d, {
      x: x + 0.28, y: 3.15, w: 3.15, h: 1.7,
      fontFace: FONT.body, fontSize: 15, color: COLOR.body, margin: 0,
    });
  });

  s.addText("Talking point: the archive is heterogeneous. Analysis starts after we intersect member, time, and grid — not from the 75 × 1,103 checklist.", {
    x: MX, y: 5.55, w: 11.6, h: 0.95,
    fontFace: FONT.body, fontSize: 15, italic: true, color: COLOR.body, margin: 0,
  });
}

pres.writeFile({
  fileName: "/Users/mimi/Documents/Code/Github/GSA-work/explore/sythetic/case/caseA/other/CMIP6_catalog_not_aligned.pptx",
}).then(() => console.log("wrote pptx"));
