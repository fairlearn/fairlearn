import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from "office-ui-fabric-react";

export interface IAccuracyTabStyles {
    itemCell: IStyle;
    iconClass: IStyle;
    itemsList: IStyle;
    frame: IStyle;
    main: IStyle;
    header: IStyle;
    textBody: IStyle;
}

export const AccuracyTabStyles: () => IProcessedStyleSet<IAccuracyTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IAccuracyTabStyles>({
        itemCell: [
            {
              padding: "30px 36px 20px 0",
              width: "100%",
              position: "relative",
              float: "left",
              cursor: "pointer",
              boxSizing: "border-box",
              borderBottom: "1px solid #CCCCCC",
              selectors: {
                '&:hover': { background: "lightgray" }
              }
            }
          ],
          iconClass: {
              fontSize: "20px",
              position: "absolute",
              right: "10px",
              top: "10px"
          },
          itemsList: {
              overflowY: "auto"
          },
          frame: {
              height: "100%",
          },
          main: {
              height: "100%",
              maxWidth: "750px",
              flex: 1
          },
          header: {
              color: "#333333",
              fontSize: "32px",
              lineHeight: "39px",
              fontWeight: "100",
              margin: "26px 0"
          },
          textBody: {
              paddingTop: "12px",
              fontSize: "18px",
              lineHeight: "24px",
              fontWeight: "300",
              paddingBottom: "12px"
          }
    });
};