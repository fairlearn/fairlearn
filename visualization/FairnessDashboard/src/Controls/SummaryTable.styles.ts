import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights } from "office-ui-fabric-react";

export interface ISummaryTableStyles {
    minMaxLabel: IStyle;
    groupCol: IStyle;
    groupLabel: IStyle;
    flexCol: IStyle;
    binBox: IStyle;
    binTitle: IStyle;
    metricCol: IStyle;
    metricBox: IStyle;
    metricLabel: IStyle;
    frame: IStyle;
}

export const SummaryTableStyles: () => IProcessedStyleSet<ISummaryTableStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ISummaryTableStyles>({
        minMaxLabel: {
            padding: "1px 9px",
            marginTop: "4px",
            //color: "#FFFFFF",
            color: theme.semanticColors.bodySubtext,
            //fontSize: "10px",
            //lineHeight: "20px",
            //fontWeight: "400",
            //backgroundColor: "#999999"
            backgroundColor: theme.semanticColors.bodyStandoutBackground
        },
        groupCol: {
            display: "inline-flex",
            flexDirection:"column",
            height: "100%",
            width: "max-content"
        },
        groupLabel: {
            //color: "#333333",
            color: theme.semanticColors.bodyText,
            //fontSize: "12px",
            //lineHeight: "12px",
            //fontWeight: "500",
            height: "26px"
        },
        flexCol: {
            display: "flex",
            flex: 1,
            flexDirection: "column",
            borderTop: "1px solid",
            borderBottom: "1px solid",
            borderColor: theme.semanticColors.bodyDivider //#CCCCCC
            
        },
        binBox: {
            flex: 1,
            // minWidth: "100px",
            // maxWidth: "200px",
            // width: "max-content",
            width: "130px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            borderBottom: "0.5px dashed",
            borderColor: theme.semanticColors.bodyDivider //#CCCCCC
        },
        binTitle: {
            //color: "#333333",
            color: theme.semanticColors.bodyText,
            //fontSize: "15px",
            //lineHeight: "18px",
            //fontWeight: "400"
        },
        metricCol: {
            display: "inline-flex",
            flexDirection:"column",
            height: "100%",
            width: "120px"
        },
        metricLabel: {
            //color: "#333333",
            color: theme.semanticColors.bodyText,
            //fontSize: "12px",
            //lineHeight: "12px",
            //fontWeight: "500",
            height: "26px",
            paddingLeft: "10px"
        },
        metricBox: {
            flex: 1,
            paddingLeft: "10px",
            //color: "#333333",
            color: theme.semanticColors.bodyText,
            //fontSize: "32px",
            //lineHeight: "39px",
            fontWeight: FontWeights.light,
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            borderLeft: "0.5px dashed",
            borderBottom: "0.5px dashed",
            borderColor: theme.semanticColors.bodyDivider //#CCCCCC

        },
        frame: {
            paddingBottom: "19px",
            display: "flex"
        }
    });
};