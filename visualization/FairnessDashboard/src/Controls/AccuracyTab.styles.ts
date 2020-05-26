import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from "office-ui-fabric-react";

export interface IAccuracyTabStyles {
    //itemCell: IStyle;
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
        // itemCell: [
        //     {
        //       padding: "30px 36px 20px 0",
        //       width: "100%",
        //       position: "relative",
        //       float: "left",
        //       cursor: "pointer",
        //       boxSizing: "border-box",
        //       borderBottom: "1px solid",
        //       borderColor: "#CCCCCC",
        //       selectors: {
        //         '&:hover': { background: "lightgray" }
        //       }
        //     }
        //   ],
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
              //color: "#333333",
              color: theme.semanticColors.bodyText,
              //fontSize: "32px",
              fontSize: FontSizes.xxLargePlus,
              //lineHeight: "39px",
              //fontWeight: "100",
              fontWeight: FontWeights.light,
              margin: "26px 0"
          },
          textBody: {
              paddingTop: "12px",
              //fontSize: "18px",
              //fontSize: FontSizes.large,
              //lineHeight: "24px",
              //fontWeight: "300",
              fontWeight: FontWeights.semilight,
              color: theme.semanticColors.bodySubtext,
              paddingBottom: "12px"
          }
    });
};