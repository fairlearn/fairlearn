import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from "office-ui-fabric-react";

export interface IFairnessWizardStyles {
    frame: IStyle;
    thinHeader: IStyle;
    headerLeft: IStyle;
    //headerRight: IStyle;
    pivot: IStyle;
    body: IStyle;
    errorMessage: IStyle;
}

export const FairnessWizardStyles: () => IProcessedStyleSet<IFairnessWizardStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IFairnessWizardStyles>({
        frame: {
            minHeight: "800px",
            minWidth: "800px",
            //fontFamily: `"Segoe UI", "Segoe UI Web (West European)", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", sans-serif`
        },
        thinHeader: {
            height: "36px",
            // #333333
            backgroundColor: theme.palette.neutralPrimary,
            color: theme.palette.white
        },
        headerLeft: {
            //fontSize: "15px",
            //lineHeight: "24px",
            //fontWeight: "500",
            padding: "20px"
        },
        // headerRight: {
        //     fontSize: "12px",
        //     padding: "20px"
        // },
        pivot: {
            flex: 1,
            display: "flex",
            flexDirection: "column",
            //backgroundColor: "#F2F2F2",
            //backgroundColor: theme.semanticColors.bodyBackgroundHovered,
            padding: "30px 90px 0 82px"
        },
        body: {
            flex: 1,
            display: "flex",
            flexDirection: "column",
            backgroundColor: theme.semanticColors.bodyBackground
        },
        errorMessage: {
            padding: "50px",
            fontSize: "18px"
        }
    });
};